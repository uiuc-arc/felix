import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, optim
from tqdm import tqdm, trange
from tvm.auto_scheduler.cost_model import MLPCostModel

from .features import TorchFeatures
from .sym_task import ConfigInfo, SymTask, TaskInstance, TaskPerfFunc, print_perf

_logger = logging.getLogger(__name__)
__all__ = ["Optimizer"]
PathLike = Union[Path, str]


class Optimizer:
    def __init__(self, tasks: List[Tuple[SymTask, TaskInstance]], cost_model: MLPCostModel) -> None:
        self.timer = Timer()
        self.tasks: Dict[int, TaskPerfFunc] = {}
        for sym_task, inst in tqdm(tasks):
            with self.timer.time(f"create[{inst.idx}]"):
                task_f = TaskPerfFunc(sym_task, inst.sizes, inst.weight, cost_model)
            self.tasks[inst.idx] = task_f

    def tune_one_round(
        self,
        lr: float,
        n_seed_configs: int,
        n_optim_steps: int,
        n_measurements: int,
        log_file: Optional[PathLike] = None,
    ):
        results = {}
        for idx, task_perf_f in self.tasks.items():
            optimizer = TaskOptimizer(task_perf_f, lr, n_optim_steps, n_seed_configs)
            _logger.info(f"Tuning for task {idx}")
            with self.timer.time(f"tune[{idx}]"):
                for _ in trange(n_optim_steps):
                    optimizer.optimize_step()
                configs = optimizer.get_best_configs()
            _logger.info("Measuring latency of best configs empirically...")
            task_f = optimizer.task_f
            with self.timer.time(f"measure[{idx}]"):
                mcs = task_f.measure_configs_latency_(configs[:n_measurements])
            assert idx not in results
            results[idx] = task_f.weight, configs
            # Print only measured configs one by one.
            print_configs(idx, task_f.weight, mcs, True)
        total_lat = 0
        configs = []
        for idx in self.tasks:
            weight, configs_ = results[idx]
            best_perf = print_configs(idx, weight, configs_, False)
            if best_perf is not None:
                total_lat += best_perf.latency_us * weight
            configs += configs_
        if log_file is not None:
            output_json_lines = [c.to_tvm_string() for c in configs]
            with open(log_file, "w") as f:
                f.writelines(output_json_lines)
        _logger.info("Total latency: %.2f us", total_lat)
        self.timer.log_all()


def print_configs(idx: int, weight: int, configs: List[ConfigInfo], print_each_config: bool):
    m_perf = [perf for c in configs if (perf := c.get_measured_perf()) is not None]
    best_m_perf = max(m_perf, default=None)
    best_p_perf = max([c.pred_perf for c in configs], default=None)
    first_line = (
        f"Task {idx}: best measured {print_perf(best_m_perf, weight)}, "
        f"best predicted {print_perf(best_p_perf, weight)}. "
    )
    if print_each_config:
        first_line += "Top configs:"
        printed_configs = [
            f"  pred perf={print_perf(config.pred_perf, weight)}, "
            f"measured={print_perf(config.get_measured_perf(), weight)}, \n"
            f"  config: {config.config} (from {config.sketch_f.sketch.backbone})"
            for config in configs
        ]
    else:
        printed_configs = []
    logger_str = "\n".join([first_line] + printed_configs)
    _logger.info(logger_str)
    return best_m_perf


class TaskOptimizer:
    def __init__(
        self,
        task_f: TaskPerfFunc,
        lr: float,
        n_iters: int,
        n_configs: int,
        constraint_k: int = 5,
    ) -> None:
        self.cost_mag = 1
        self.constraint_k = constraint_k
        self.task_f = task_f
        self.configs = task_f.rand_configs(n_configs)
        self.optim = optim.Adam(self.configs, lr=lr)
        self.lr_sched = optim.lr_scheduler.MultiStepLR(
            self.optim, milestones=[n_iters // 2], gamma=0.1
        )
        # self._configs: [n_steps] x [n_sketches] x [n_configs, n_params]
        self._configs: List[List[Tensor]] = []
        # self._costs: [n_steps] x [n_sketches, n_configs]
        self._costs: List[Tensor] = []
        # constraints: [n_steps] x [n_sketches] x [n_configs, n_constraints]
        self._constraints: List[List[Tensor]] = []
        self._step_idx = 0

    def optimize_step(self):
        # 1. Compute loss.
        # costs: [n_sketches, n_configs]
        # constraints: [n_sketches] x [n_configs, n_constraints]
        perfs, constraints = self.task_f.forward(self.configs)
        if self._step_idx % 10 == 0:
            min_, max_ = (
                perfs.min(dim=1).values.detach(),
                perfs.max(dim=1).values.detach(),
            )
            _logger.info(f"min perf={min_}, max perf={max_}")
        self._step_idx += 1
        # Model predicts a performance score (higher is better)
        # Need to invert it into something lower-better.
        costs = -perfs
        # Minimizing `costs` but map the costs to roughly have a magnitude of 1.
        lat_loss = torch.sum(costs / self.cost_mag)
        # Constraints are good when <= 0.
        penalties = sum([(cs.relu() ** 2).sum() for cs in constraints])
        loss = lat_loss + self.constraint_k * penalties
        # 2. Keep a history of these values.
        self._costs.append(costs.clone().detach())
        self._configs.append([c.clone().detach() for c in self.configs])
        self._constraints.append([c.clone().detach() for c in constraints])
        # 3. Backprop and update.
        loss.backward()
        for config, sketch in zip(self.configs, self.task_f.sketches):
            self._fix_nan_grads(sketch.features, config)
        self.optim.step()
        self.optim.zero_grad()
        self.lr_sched.step()

    def get_history(self):
        # configs: [n_sketches] x [n_steps, n_configs, n_params]
        configs = [torch.stack(cs, dim=0) for cs in zip(*self._configs)]
        # costs: [n_sketches, n_steps, n_configs]
        costs = torch.stack(self._costs, dim=0).transpose(0, 1)
        # constraints: [n_sketches] x [n_steps, n_configs, n_constraints]
        constraints = [torch.stack(cs, dim=0) for cs in zip(*self._constraints)]
        return configs, costs, constraints

    def get_best_configs(self):
        # configs: [n_sketches] x [n_steps x n_configs, n_params]
        configs = [torch.cat(cs, dim=0).round() for cs in zip(*self._configs)]
        return self.task_f.get_best_configs(configs)

    def _fix_nan_grads(self, config_gen: TorchFeatures, conf: Tensor):
        grad = conf.grad
        assert grad is not None
        nan_grad = torch.any(torch.isnan(grad).flatten(1, -1), dim=1)
        n_configs = int(nan_grad.count_nonzero().item())
        if n_configs == 0:
            return
        _logger.info(f"Fixing {n_configs} NaN gradients")
        with torch.no_grad():
            conf[nan_grad] = config_gen.rand_configs(n_configs)
            grad[nan_grad] = 0
            optim_state = self.optim.state[conf]
            for k in ("exp_avg", "exp_avg_sq"):
                buf = optim_state.get(k)
                if buf is not None:
                    buf[nan_grad] = 0


class Timer:
    def __init__(self) -> None:
        self.timings: Dict[str, float] = {}

    def time(self, task_name: str):
        return TimerGuard(task_name, self.timings)

    def log_all(self):
        _logger.info(
            "\n".join(
                f"{task_name}: {duration:.3f}s" for task_name, duration in self.timings.items()
            )
        )


class TimerGuard:
    def __init__(self, task_name: str, results: dict) -> None:
        self.task_name = task_name
        self.results = results
        self.tic = None

    def __enter__(self):
        self.tic = time.time()
        return self

    def __exit__(self, *args):
        assert self.tic is not None
        toc = time.time()
        duration = toc - self.tic
        if self.task_name in self.results:
            self.results[self.task_name] += duration
        else:
            self.results[self.task_name] = duration
