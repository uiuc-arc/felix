import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import torch
from torch import Tensor, nn, optim
from torch.optim import lr_scheduler as lrs
from tqdm import tqdm, trange
from tvm import auto_scheduler as ansor
from tvm.auto_scheduler.cost_model import DatasetBuilder, LatAndThruput, Performance
from tvm.auto_scheduler.measure_record import dump_record_to_string

from . import ffi
from .cost_model import MLPModelPLWrapper
from .sym_task import Sketch, SymTask, TaskInstance
from .utils import make_runner, transpose2

_logger = logging.getLogger(__name__)
__all__ = ["Optimizer"]
PathLike = Union[Path, str]


class Optimizer:
    def __init__(
        self, tasks: List[Tuple[SymTask, TaskInstance]], perf_model: MLPModelPLWrapper
    ) -> None:
        self.timer = Timer()
        self.tasks: Dict[int, Tuple[TaskPerfFunc, int]] = {}
        for sym_task, inst in tqdm(tasks):
            with self.timer.time(f"create[{inst.idx}]"):
                task_f = TaskPerfFunc(sym_task, inst.sizes, perf_model)
            self.tasks[inst.idx] = task_f, inst.weight
        self.perf_model = perf_model

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
            _logger.info(f"Task {idx}: best predicted {best_pred_perf} (weight = {weight})")
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
                    f"Overall: best measured {best_actual}, best predicted {best_pred} (weight = {weight})"
                )
        self._summarize(results, log_file)

    def _summarize(self, results: dict, log_file):
        total_lat = 0
        configs = []
        for idx in self.tasks:
            configs_ = results[idx]
            weight = self.tasks[idx][1]
            best_actual, best_predicted = get_best_configs(configs_)
            _logger.info(
                f"Task {idx}: best measured {best_actual}, best predicted {best_predicted} (weight = {self.tasks[idx][1]})"
            )
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
        self.perf_model.train_self(
            builder.to_dataset(),
            batch_size=32,
            n_epoch=30,
            early_stop=5,
            lr=1e-4,
        )
        best_actual, best_pred = get_best_configs(measured_top)
        _logger.info(
            f"This round: best measured {best_actual}, best predicted {best_pred} (weight = {weight})"
        )
        return measured_top


class TaskOptimizer:
    CK = 1

    def __init__(
        self,
        task_f: "TaskPerfFunc",
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
        # perfs: [n_sketches, n_configs]
        # constraints: [n_sketches] x [n_configs, n_constraints]
        perfs, constraints = self.task_f.forward(self.configs)
        if self._step_idx % 10 == 0:
            to_perf = lambda x: self.task_f.perf_model.output_to_performance(self.task_f.flops, x)
            min_perfs = [to_perf(p) for p in perfs.min(dim=1).values.tolist()]
            max_perfs = [to_perf(p) for p in perfs.max(dim=1).values.tolist()]
            _logger.info(f"min perf={min_perfs}, max perf={max_perfs}")
        self._step_idx += 1
        # Constraints are good when <= 0.
        penalties = [(cs.relu() ** 2).sum() for cs in constraints]
        # Model predicts a performance score (higher is better)
        # Need to invert it into something lower-better.
        loss = torch.sum(-perfs) + self.CK * sum(penalties)
        # 2. Keep a history of these values.
        self._configs.append([c.clone().detach() for c in self.configs])
        # 3. Backprop and update.
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        if self.lr_sched is not None:
            self.lr_sched.step()


class SingleRoundTaskOptimizer(TaskOptimizer):
    def __init__(self, task_f: "TaskPerfFunc", n_seeds: int, n_steps: int, lr: float) -> None:
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
        task_f: "TaskPerfFunc",
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
            builder.add_configs_(features, self.task_f.flops, lat_secs, {})
        return cis[:n_top]  # Only return the top configs.


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

    def to_tvm_string(self):
        mres = self.measure_result
        if mres is None:
            mres = ansor.MeasureResult([], 0, "", 0, 0)
        return dump_record_to_string(self.get_measure_input(), mres)


def get_best_configs(configs: List[ConfigInfo]):
    m_perf = [perf for c in configs if (perf := c.get_measured_perf()) is not None]
    best_m_perf = max(m_perf, default=None)
    best_p_perf = max([c.pred_perf for c in configs], default=None)
    return best_m_perf, best_p_perf


def measure_configs_latency_(configs: List[ConfigInfo]):
    from tvm.auto_scheduler.measure import ProgramMeasurer

    measurer = ProgramMeasurer(ansor.LocalBuilder(), make_runner(), [], False)
    results = ffi.measure_performance(measurer, [c.get_measure_input() for c in configs])
    for c, result in zip(configs, results):
        if result.error_no != 0:
            _logger.warning(result.error_msg)
        c.measure_result = result
    return configs


class TaskPerfFunc(nn.Module):
    def __init__(self, task: SymTask, sizes: Dict[str, int], perf_model: MLPModelPLWrapper) -> None:
        from .sym_dag import Conv

        super().__init__()
        self.sym_task = task
        self.sizes = sizes
        self.ansor_task, _ = task.make_concrete_task(sizes)

        # HACK: remove first sketch of Conv2/3d as its performance tends to be overestimated
        # and can actually be selected instead of the second sketch when it should not.
        has_conv = any(isinstance(n, Conv) for n in task.dag_nodes())
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

    def get_best_configs(
        self, configs: List[Tensor], remove_invalid: bool = True, dedup_set: Optional[set] = None
    ):
        """For each config, get its best version across all sketches.

        `configs` is a list of configs and should have the same length as
        what `rand_configs` returns (i.e., one Tensor per sketch).
        """

        # [n_sketches, n_configs]; [n_sketches] x [n_configs, n_constraints]
        n_invalid, n_dup = 0, 0
        perfs, constraints = self.forward(configs)
        max_perfs, best_sketch_ids = perfs.max(dim=0)
        ret: List[ConfigInfo] = []
        dedup_set = dedup_set or set()
        for idx, sketch in enumerate(self.sketches):
            configs_ = configs[idx]
            nan_or_inf = torch.any(torch.isnan(configs_) | torch.isinf(configs_), dim=1)
            violates_cons = torch.sum(torch.relu(constraints[idx]) ** 2, dim=1) > 0
            invalid = nan_or_inf | violates_cons
            sketch_mask = best_sketch_ids == idx
            if remove_invalid:
                n_invalid += torch.sum(sketch_mask & invalid).item()
                sketch_mask &= ~invalid
            for conf_i in torch.nonzero(sketch_mask).squeeze(1):
                config_dict = sketch.features.inv_transform_config(configs_[conf_i])
                config_kvs = tuple(sorted(config_dict.items(), key=lambda x: x[0]))
                if config_kvs in dedup_set:
                    n_dup += 1
                    continue
                dedup_set.add(config_kvs)
                perf = self.perf_model.output_to_performance(self.flops, max_perfs[conf_i].item())
                ret.append(ConfigInfo(config_dict, sketch, perf))
        _logger.info(
            f"get_best_configs: n_invalid={n_invalid}, n_dup={n_dup}, n_configs={len(ret)}"
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
