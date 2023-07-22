import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, optim
from torch.optim import lr_scheduler as lrs
from tqdm import tqdm, trange
from tvm.auto_scheduler.cost_model import DatasetBuilder
from tvm.felix.sym_task import TaskPerfFunc

from .cost_model import MLPModelPLWrapper
from .sym_task import (
    ConfigInfo,
    SymTask,
    TaskInstance,
    TaskPerfFunc,
    measure_configs_latency_,
    print_perf,
)

_logger = logging.getLogger(__name__)
__all__ = ["Optimizer"]
PathLike = Union[Path, str]


class Optimizer:
    def __init__(
        self, tasks: List[Tuple[SymTask, TaskInstance]], cost_model: MLPModelPLWrapper
    ) -> None:
        self.timer = Timer()
        self.tasks: Dict[int, Tuple[TaskPerfFunc, int]] = {}
        for sym_task, inst in tqdm(tasks):
            with self.timer.time(f"create[{inst.idx}]"):
                task_f = TaskPerfFunc(sym_task, inst.sizes, cost_model)
            self.tasks[inst.idx] = task_f, inst.weight
        self.cost_model = cost_model

    def tune_one_round(
        self,
        n_configs: int,
        n_optim_steps: int,
        n_measurements: int,
        log_file: Optional[PathLike] = None,
        lr: float = 0.5,
    ):
        results = {}
        for idx, (task_perf_f, weight) in self.tasks.items():
            optimizer = SingleRoundTaskOptimizer(task_perf_f, n_configs, n_optim_steps, lr)
            _logger.info(f"Tuning for task {idx}")
            with self.timer.time(f"tune[{idx}]"):
                for _ in trange(n_optim_steps):
                    optimizer.optimize_step()
                configs = optimizer.get_best_configs()
            _logger.info("Measuring latency of best configs empirically...")
            assert idx not in results
            results[idx] = configs
            best_pred_perf = max([c.pred_perf for c in configs], default=None)
            _logger.info(f"Task {idx}: best predicted {print_perf(best_pred_perf, weight)}")
        to_measure = [c for task_idx in self.tasks for c in results[task_idx][:n_measurements]]
        with self.timer.time(f"measure"):
            measure_configs_latency_(to_measure)
        self._summarize(results, log_file)
        return results

    def tune_multi_rounds(
        self,
        n_configs: int,
        n_first_round_steps: int,
        n_latter_round_steps: int,
        n_rounds: int,
        measure_per_round: int,
        log_file: Optional[PathLike] = None,
    ):
        results = {}
        builder = DatasetBuilder()
        n1, n2, n3 = n_first_round_steps, n_latter_round_steps, measure_per_round
        for idx, (task_perf_f, weight) in self.tasks.items():
            results[idx] = results_ = []
            optimizer = MultiRoundTaskOptimizer(task_perf_f, n_configs, n1, n2)
            for round in range(0, n_rounds):
                _logger.info(f"Round {round}:")
                if round == 0:
                    mtop = self._do_one_task_one_round(optimizer, builder, weight, n1, n3, 0)
                else:
                    mtop = self._do_one_task_one_round(
                        optimizer, builder, weight, n2, n3 // 2, n3 - n3 // 2
                    )
                results_.extend(mtop)
                best_actual, best_pred = get_best_configs(results_)
                _logger.info(
                    f"Overall: best measured {print_perf(best_actual, weight)}, "
                    f"best predicted {print_perf(best_pred, weight)}"
                )
        self._summarize(results, log_file)

    def _summarize(self, results: dict, log_file):
        total_lat = 0
        configs = []
        for idx in self.tasks:
            configs_ = results[idx]
            weight = self.tasks[idx][1]
            best_actual, _ = get_best_configs(configs_)
            _logger.info(f"Task {idx}: best measured {print_perf(best_actual, self.tasks[idx][1])}")
            if best_actual is not None:
                total_lat += best_actual.latency_us * weight
            configs += configs_
        if log_file is not None:
            output_json_lines = [c.to_tvm_string() for c in configs]
            with open(log_file, "w") as f:
                f.writelines(output_json_lines)
        _logger.info("Total latency: %.2f us", total_lat)

    def _do_one_task_one_round(
        self,
        optimizer: "MultiRoundTaskOptimizer",
        builder: DatasetBuilder,
        weight: int,
        n_steps: int,
        measure_top: int,
        measure_rand: int,
    ):
        for _ in trange(n_steps):
            optimizer.optimize_step()
        measured_top = optimizer.measure_for_dataset(measure_top, measure_rand, builder)
        self.cost_model.train_self(
            builder.to_dataset(),
            self.cost_model.main_loss,
            lr=1e-4,
            weight_decay=1e-6,
            batch_size=32,
            n_epoch=30,
            early_stop=5,
        )
        best_actual, best_pred = get_best_configs(measured_top)
        _logger.info(
            f"This round: best measured {print_perf(best_actual, weight)}, "
            f"best predicted {print_perf(best_pred, weight)}"
        )
        return measured_top


def get_best_configs(configs: List[ConfigInfo]):
    m_perf = [perf for c in configs if (perf := c.get_measured_perf()) is not None]
    best_m_perf = max(m_perf, default=None)
    best_p_perf = max([c.pred_perf for c in configs], default=None)
    return best_m_perf, best_p_perf


def print_each_config(weight: int, configs: List[ConfigInfo]):
    printed_configs = ["Top configs:"] + [
        f"  pred perf={print_perf(config.pred_perf, weight)}, "
        f"measured={print_perf(config.get_measured_perf(), weight)}, \n"
        f"  config: {config.config} (from {config.sketch_f.sketch.backbone})"
        for config in configs
    ]
    _logger.info("\n".join(printed_configs))


class TaskOptimizer:
    CK = 5

    def __init__(
        self,
        task_f: TaskPerfFunc,
        configs: List[Tensor],
        optim: optim.Optimizer,
        lr_sched: Optional[lrs._LRScheduler],
    ) -> None:
        self.task_f = task_f
        self.configs = configs
        self.optim = optim
        self.lr_sched = lr_sched
        # self._configs: [n_steps] x [n_sketches] x [n_configs, n_params]
        self._configs: List[List[Tensor]] = []
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
        lat_loss = torch.sum(costs)
        # Constraints are good when <= 0.
        penalties = sum([(cs.relu() ** 2).sum() for cs in constraints])
        loss = lat_loss + self.CK * penalties
        # 2. Keep a history of these values.
        self._configs.append([c.clone().detach() for c in self.configs])
        # 3. Backprop and update.
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        if self.lr_sched is not None:
            self.lr_sched.step()


class SingleRoundTaskOptimizer(TaskOptimizer):
    def __init__(self, task_f: TaskPerfFunc, n_seeds: int, n_steps: int, lr: float) -> None:
        configs = task_f.rand_configs(n_seeds)
        optimizer = optim.Adam(configs, lr)
        lr_sched = lrs.MultiStepLR(optimizer, milestones=[int(n_steps * 0.6)], gamma=0.2)
        super().__init__(task_f, configs, optimizer, lr_sched)

    def get_best_configs(self):
        # configs: [n_sketches] x [n_steps x n_configs, n_params]
        configs = [torch.cat(cs, dim=0).round() for cs in zip(*self._configs)]
        return self.task_f.get_best_configs(configs)


class MultiRoundTaskOptimizer(TaskOptimizer):
    def __init__(
        self,
        task_f: TaskPerfFunc,
        n_seeds: int,
        n_first_round_steps: int,
        n_latter_round_steps: int,
    ) -> None:
        configs = task_f.rand_configs(n_seeds)
        optimizer = optim.Adam(configs, lr=0.5)
        lr_sched = lrs.MultiStepLR(
            optimizer, milestones=[int(n_first_round_steps * 0.6)], gamma=0.2
        )
        super().__init__(task_f, configs, optimizer, lr_sched)
        self.n_first_round_steps = n_first_round_steps
        self.n_latter_round_steps = n_latter_round_steps
        self._dedup_set = set()

    def get_best_configs(self):
        # configs: [n_sketches] x [n_steps x n_configs, n_params]
        configs = [torch.cat(cs, dim=0).round() for cs in zip(*self._configs)]
        return self.task_f.get_best_configs(configs, dedup_set=self._dedup_set)

    def measure_for_dataset(self, n_top: int, n_rand: int, builder: DatasetBuilder):
        cis: List[ConfigInfo] = self.get_best_configs()
        rand_indices = (torch.randperm(len(cis) - n_top)[:n_rand] + n_top).tolist()
        cis = measure_configs_latency_(cis[:n_top] + [cis[i] for i in rand_indices])
        configs_by_sketch = defaultdict(list)
        for ci in cis:
            perf = ci.get_measured_perf()
            if perf is None:
                continue
            configs_by_sketch[ci.sketch_f].append((ci.config, perf.latency_us / 1e6))
        for sketch_f, conf_perfs in configs_by_sketch.items():
            configs, lat_secs = zip(*conf_perfs)
            lat_secs = torch.tensor(list(lat_secs))
            features, _ = sketch_f.features.run_on_initial_configs(configs)
            builder.add_configs_(features, self.task_f.flops, lat_secs)
        return cis[:n_top]  # Only return the top configs.


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
