import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, cast

import torch
from torch import Tensor, nn, optim
from torch.optim import lr_scheduler as lrs
from tvm import auto_scheduler as ansor
from tvm.auto_scheduler.cost_model import DatasetBuilder, LatAndThruput, Performance
from tvm.auto_scheduler.measure_record import save_records

from . import ffi
from .cost_model import MLPModelPLWrapper
from .sym_task import Sketch, SymTask, TaskInstance
from .utils import make_runner, transpose2

_logger = logging.getLogger(__name__)
__all__ = ["Optimizer"]
PathLike = Union[Path, str]


class Optimizer:
    ALPHA, BETA = 0.2, 2
    BW_WINDOW_SIZE = 3

    def __init__(
        self, tasks: List[Tuple[SymTask, TaskInstance]], perf_model: MLPModelPLWrapper
    ) -> None:
        self.timer = Timer()
        _logger.info("Extracting features from tasks...")
        tasks = sorted(tasks, key=lambda x: x[1].idx)
        self.tasks = [
            (TaskPerfFunc(sym, inst.sizes, perf_model), inst.weight, inst.idx)
            for sym, inst in tasks
        ]
        self.perf_model = perf_model
        self.n_measure, self.n_rounds = 0, 0
        self.data_builder = DatasetBuilder()
        self.output_file: Path | None = None

    def tune(
        self,
        n_measurements: int,
        n_config_seeds: int,
        n_grad_steps: int,
        measure_per_round: int,
        log_file: Optional[PathLike] = None,
    ):
        def lr_sched_f(optim):
            return lrs.MultiStepLR(optim, milestones=[int(n_grad_steps * 0.3)], gamma=0.2)

        optims = [
            SingleTaskOptimizer(perf_f, n_config_seeds, lr_sched_f) for perf_f, _, _ in self.tasks
        ]
        if log_file is not None:
            self.output_file = Path(log_file)
        # A round of warmup for each task
        for idx in range(len(self.tasks)):
            self._do_one_task_one_round(idx, optims[idx], n_grad_steps, measure_per_round, 0)
        self._print_total_latency(optims)
        # Keep tuning till we run out of budget
        while self.n_measure < n_measurements:
            next_task = self._gradient_sched(optims)
            self._do_one_task_one_round(
                next_task, optims[next_task], n_grad_steps // 4, measure_per_round, 0
            )
            self._print_total_latency(optims)
        self._summarize(optims, log_file)

    def _print_total_latency(self, optims):
        best_costs = torch.tensor([optim.least_lat_history[-1] for optim in optims])
        if torch.any(best_costs == 1e10):
            return None
        weights = torch.tensor([weight for _, weight, _ in self.tasks])
        if (tlat := (best_costs * weights).sum().item()) is not None:
            _logger.info(f"Network total latency: {tlat * 1e6:.2f} us")

    def _summarize(self, optims: List["SingleTaskOptimizer"], log_file: Optional[PathLike]):
        total_lat = 0
        for idx in range(len(self.tasks)):
            optim = optims[idx]
            _, weight, idx_ = self.tasks[idx]
            best = optim.least_lat_history[-1] * 1e6
            _logger.info(f"Task {idx_}: {best} us * {weight}")
            total_lat += best * weight
        _logger.info("Total latency: %.2f us", total_lat)

    def _do_one_task_one_round(
        self, task_idx: int, optimizer, n_steps: int, n_top: int, n_rand: int, train: bool = True
    ):
        _, weight, idx = self.tasks[task_idx]
        _logger.info(f"Round {self.n_rounds} (task {idx}):")
        # train performance model
        if not train:
            _logger.info("Skipping training")
        elif len(self.data_builder) < 128 * 2:
            _logger.info(f"Dataset size {len(self.data_builder)} is too small, skipping training")
        else:
            dataset = self.data_builder.to_dataset()
            if self.output_file is not None:
                torch.save(dataset, self.output_file.with_suffix(".pkl"))
            n_batches = int(30000 / len(dataset))
            n_early_stop = n_batches // 5
            self.perf_model.train_self(dataset, 32, n_batches, n_early_stop, 1e-4)
        sorted_configs = optimizer.optimize_round(n_steps)
        measured = optimizer.measure_configs(sorted_configs, n_top, n_rand, self.data_builder)
        if self.output_file is not None:
            mis = [c.get_measure_input() for c in measured]
            mrs = [c.get_measure_result() for c in measured]
            save_records(str(self.output_file), mis, mrs)
        best_actual, best_pred = get_best_configs(measured)
        self.n_measure += len(measured)
        self.n_rounds += 1
        _logger.info(
            f"Task {idx}: best measured {best_actual}, best predicted {best_pred} (weight = {weight})"
            f"\n  task history best {optimizer.least_lat_history[-1] * 1e6:.2f} us"
            f"\n  {self.n_measure} measurements done"
        )
        return measured

    def _gradient_sched(self, optims: List["SingleTaskOptimizer"]) -> int:
        def bgrad(hist):
            if len(hist) >= self.BW_WINDOW_SIZE + 1:
                return (hist[-1] - hist[-1 - self.BW_WINDOW_SIZE]) / self.BW_WINDOW_SIZE
            else:
                return 0

        n = len(self.tasks)
        gradients = torch.zeros(n)
        # for idx in self.dead_tasks:
        #     gradients[idx] = 0
        best_costs = torch.tensor([optim.least_lat_history[-1] for optim in optims])
        task_rounds = torch.tensor([len(optim.least_lat_history) for optim in optims])
        # compute gradient from chain rule : (delta f / delta g_i)
        chain_grad = torch.tensor([weight for _, weight, _ in self.tasks])
        # compute (g_i(t_i) - g(t_i - \Delta t)) / (\Delta t)
        backward_grad = torch.tensor([bgrad(optim.least_lat_history) for optim in optims])
        # compute (g_i(t_i + \Delta t) - g(t_i)) / (\Delta t)
        g_next_1 = best_costs - (best_costs / task_rounds)
        g_next_2 = torch.ones(n) * self.BETA * 1e30  # FIXME: g_next_2 disabled for now
        forward_grad = torch.minimum(g_next_1, g_next_2) - best_costs
        gradients = chain_grad * (self.ALPHA * backward_grad + (1 - self.ALPHA) * forward_grad)
        assert torch.all(gradients <= 0)
        # group_id = self.tag_to_group_id.get(self.task_tags[i], None)
        # if group_id is not None and len(self.group_task_ids[group_id]) > 1:
        #     best_flops = max(
        #         [self.flop_cts[j] / best_costs[j] for j in self.group_task_ids[group_id]]
        #     )
        #     g_next_2 = self.beta * self.flop_cts[i] / best_flops
        if gradients.max() == gradients.min():
            return int(torch.randint(0, n, (1,)).item())
        else:
            return int(torch.argmin(gradients).item())


class SingleTaskOptimizer:
    GRAD_RATIO_TARGET = 5

    def __init__(self, task_f: "TaskPerfFunc", n_seeds: int, lr_sched_f: Callable) -> None:
        self.task_f = task_f
        self.n_seeds = n_seeds
        self.configs = task_f.rand_configs(n_seeds)
        self.optim = optim.Adam(self.configs, lr=0.5)
        self.lr_sched = lr_sched_f(self.optim)
        self.least_lat_history = []
        self._dict_conf_hist = set()

    def optimize_round(self, n_steps: int):
        for c0, c1 in zip(self.configs, self.task_f.rand_configs(self.n_seeds)):
            c0.data = c1.data.clone()
        # conf_hist: [n_steps] x [n_sketches] x [n_configs, n_params]
        conf_hist: List[List[Tensor]] = []
        for step_idx in range(n_steps):
            self.optimize_step(step_idx)
            # Keep a history of config values.
            conf_hist.append([c.clone().detach() for c in self.configs])
        # configs: [n_sketches] x [n_steps x n_configs, n_params]
        configs = [torch.cat(cs, dim=0) for cs in zip(*conf_hist)]
        return self.task_f.rounded_sorted_configs(configs, dedup_set=self._dict_conf_hist)

    def optimize_step(self, step: int):
        def to_perf(x):
            return self.task_f.perf_model.output_to_performance(self.task_f.flops, x)

        # 1. Compute loss.
        # perfs: [n_sketches, n_configs]
        # constraints: [n_sketches] x [n_configs, n_constraints]
        perfs, constraints = self.task_f.forward(self.configs)
        if step % 20 == 0:
            min_perfs = [to_perf(p) for p in perfs.min(dim=1).values.tolist()]
            max_perfs = [to_perf(p) for p in perfs.max(dim=1).values.tolist()]
            _logger.info(f"min perf={min_perfs}, max perf={max_perfs}")
        # Constraints are good when <= 0.
        penalties = [(cs.relu() ** 2).sum() for cs in constraints]
        # 2. Backprop and update.
        self.optim.zero_grad()
        for config, perf, penalty in zip(self.configs, perfs, penalties):
            # Model predicts a performance score (higher is better)
            # Need to invert it into something lower-better.
            cost_grads = torch.autograd.grad(-perf.sum(), config, retain_graph=True)[0]
            penalty_grads = torch.autograd.grad(penalty, config, retain_graph=True)[0]
            pgrad_norm = penalty_grads.norm(dim=1)
            pscale = self.GRAD_RATIO_TARGET * cost_grads.norm(dim=1) / pgrad_norm
            pscale = torch.where(pgrad_norm == 0, torch.ones_like(pscale), pscale)
            grad = cost_grads + pscale.unsqueeze(1) * penalty_grads
            config.grad = grad
        self.optim.step()
        if self.lr_sched is not None:
            self.lr_sched.step()

    def measure_configs(
        self, configs: List["ConfigInfo"], n_top: int, n_rand: int, builder: DatasetBuilder
    ):
        top, rest = configs[:n_top], configs[n_top:]
        rand = [rest[i] for i in torch.randperm(len(rest))[:n_rand].tolist()]
        to_measure = top + rand
        _logger.info(f"Measuring {len(to_measure)} ({len(top)} + {len(rand)}) configs")
        measured = measure_configs_latency_(to_measure)
        # Update lowest latency
        lats = [perf.latency_s for c in configs if (perf := c.get_measured_perf()) is not None]
        this_least_lat = min(lats, default=1e10)
        prev_least_lat = self.least_lat_history[-1] if self.least_lat_history else 1e10
        self.least_lat_history.append(min(this_least_lat, prev_least_lat))
        # Add configs to dataset builder.
        configs_by_sketch = defaultdict(list)
        for config in measured:
            perf = config.get_measured_perf()
            if perf is None:
                continue
            configs_by_sketch[config.sketch_f].append((config.config, perf.latency_s))
        for sketch_f, conf_perfs in configs_by_sketch.items():
            configs, lat_secs = zip(*conf_perfs)
            lat_secs = torch.tensor(list(lat_secs))
            features, _ = sketch_f.features.run_on_initial_configs(configs)
            mask = (torch.isnan(features) | torch.isinf(features)).any(dim=1).any(dim=1)
            builder.add_configs_(features[~mask], self.task_f.flops, lat_secs[~mask], {})
        return to_measure[:n_top]  # Only return the top configs.


class SketchPerfFunc(nn.Module):
    def __init__(self, task_perf_f, sketch: Sketch, cost_f: MLPModelPLWrapper) -> None:
        super().__init__()
        self.sketch = sketch
        self.task_perf_f = task_perf_f
        self.features = sketch.fetch_features(task_perf_f.sizes)
        self.cost_f = cost_f

    def get_flops(self):
        return self.sketch.parent_task.get_flops(self.task_perf_f.sizes)

    def forward(self, configs: Tensor) -> Tuple[Tensor, Tensor]:
        feats, constraints = self.features(configs)
        perf = self.cost_f.forward(feats)
        return perf, constraints

    def make_measure_inputs(self, configs: List[Dict[str, int]]):
        task = self.task_perf_f.ansor_task
        return [
            ansor.MeasureInput(task, ffi.state_from_config(task, self.sketch.tr_steps, c))
            for c in configs
        ]


@dataclass
class ConfigInfo:
    config: Dict[str, int]
    sketch_f: SketchPerfFunc
    pred_perf: Performance
    measure_input: Optional[ansor.MeasureInput] = None
    measure_result: Optional[ansor.MeasureResult] = None

    def get_measured_perf(self) -> Optional[LatAndThruput]:
        import numpy as np

        mres = self.measure_result
        if mres is None or mres.error_no != 0:
            return None
        lat_sec = np.mean([float(x) for x in mres.costs])
        return LatAndThruput.from_latency_s(self.sketch_f.get_flops(), lat_sec)

    def get_measure_input(self):
        if self.measure_input is None:
            self.measure_input = self.sketch_f.make_measure_inputs([self.config])[0]
        return self.measure_input

    def get_measure_result(self):
        if self.measure_result is None:
            return ansor.MeasureResult([], 0, "", 0, 0)
        return self.measure_result


def get_best_configs(configs: List[ConfigInfo]):
    m_perf = [perf for c in configs if (perf := c.get_measured_perf()) is not None]
    best_m_perf = max(m_perf, default=None)
    best_p_perf = max([c.pred_perf for c in configs], default=None)
    return best_m_perf, best_p_perf


def measure_configs_latency_(configs: List[ConfigInfo]):
    from os import environ

    from tvm.auto_scheduler.measure import ProgramMeasurer

    device = int(environ.get("TVM_FELIX_DEVICE", "0"))
    _logger.info(f"Measuring on device {device}")
    measurer = ProgramMeasurer(ansor.LocalBuilder(), make_runner(device=device), [], False)
    results = ffi.measure_performance(measurer, [c.get_measure_input() for c in configs])
    for c, result in zip(configs, results):
        if result.error_no != 0:
            _logger.warning(result.error_msg)
        c.measure_result = result
    return configs


class TaskPerfFunc(nn.Module):
    def __init__(self, task: SymTask, sizes: Dict[str, int], perf_model: MLPModelPLWrapper) -> None:
        from .sym_dag import Conv, Dense, TransposeConv2d

        super().__init__()
        self.sym_task = task
        self.sizes = sizes
        self.ansor_task, _ = task.make_concrete_task(sizes)

        # HACK: remove first sketch of Conv2/3d as its performance tends to be overestimated
        # and can actually be selected instead of the second sketch when it should not.
        has_conv = any(isinstance(n, (Conv, Dense, TransposeConv2d)) for n in task.dag_nodes())
        sketches = task.sketches[1:] if has_conv else task.sketches
        sketches = [SketchPerfFunc(self, sketch, perf_model) for sketch in sketches]
        self._sketches = nn.ModuleList(sketches)
        self.perf_model = perf_model
        self.flops = task.get_flops(sizes)

    @property
    def sketches(self):
        return cast(List[SketchPerfFunc], list(self._sketches))

    def forward(self, configs: List[Tensor]):
        if len(configs) != len(self._sketches):
            raise ValueError(
                f"Number of configs {len(configs)} does not match number of sketches "
                f"{len(self._sketches)} for task {self.task}"
            )
        perfs, constraints = transpose2(
            [sketch.forward(config) for sketch, config in zip(self.sketches, configs)]
        )
        # perfs: [n_sketches] x [n_configs]
        # constraints: [n_sketches] x [n_configs, n_constraints]
        return torch.stack(perfs, dim=0), constraints

    def rand_configs(self, n: int) -> List[Tensor]:
        return [sketch.features.rand_configs(n).requires_grad_() for sketch in self.sketches]

    def rounded_sorted_configs(
        self, configs: List[Tensor], remove_invalid: bool = True, dedup_set: Optional[set] = None
    ):
        configs = [c.round() for c in configs]
        n_invalid, n_dup = 0, 0
        # [n_sketches, n_configs]; [n_sketches] x [n_configs, n_constraints]
        perfs, constraints = self.forward(configs)
        ret: List[ConfigInfo] = []
        dedup_set = dedup_set or set()
        for idx, sketch in enumerate(self.sketches):
            configs_ = configs[idx]
            nan_or_inf = torch.any(torch.isnan(configs_) | torch.isinf(configs_), dim=1)
            violates_cons = torch.sum(torch.relu(constraints[idx]) ** 2, dim=1) > 0
            invalid = nan_or_inf | violates_cons
            if remove_invalid:
                n_invalid += torch.sum(invalid).item()
                configs_ = configs_[~invalid]
            for config, perf in zip(configs_, perfs[idx]):
                config_dict = sketch.features.inv_transform_config(config)
                config_kvs = tuple(sorted(config_dict.items(), key=lambda x: x[0]))
                if config_kvs in dedup_set:
                    n_dup += 1
                    continue
                dedup_set.add(config_kvs)
                perf = self.perf_model.output_to_performance(self.flops, perf.item())
                ret.append(ConfigInfo(config_dict, sketch, perf))
        _logger.debug(
            f"get_sorted_configs: n_invalid={n_invalid}, n_dup={n_dup}, n_configs={len(ret)}"
        )
        return sorted(ret, key=lambda x: x.pred_perf, reverse=True)

    def __repr__(self) -> str:
        return (
            f"TaskLatFunc({self.sym_task}, sizes={self.sizes}," f"sketches={len(self._sketches)})"
        )

    __str__ = __repr__


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
