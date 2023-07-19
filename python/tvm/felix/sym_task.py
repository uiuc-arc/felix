import logging
import pickle as pkl
from collections import defaultdict
from dataclasses import dataclass
from functools import total_ordering
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union, cast

import onnx
import torch
from torch import Tensor, nn
from tqdm import tqdm
from tvm import auto_scheduler as ansor
from tvm.auto_scheduler.cost_model import MLPCostModel
from tvm.auto_scheduler.measure_record import dump_record_to_string
from tvm.auto_scheduler.workload_registry import register_workload_tensors

from . import ffi, utils
from .sketch import Sketch, SketchPerfFunc
from .sym_dag import RelayOpBuilder, SymbolicDAG
from .utils import HW_PARAMS, TARGET, AnsorTaskWeight, PathLike

_logger = logging.getLogger(__name__)
__all__ = [
    "SymTask",
    "TaskInstance",
    "SymTaskAndInstances",
    "batch_create_tasks",
    "extract_tasks",
    "load_and_register_tasks",
    "extract_tenset_pickle_tasks",
]


class TaskInstance(NamedTuple):
    idx: int
    weight: int
    sizes: dict
    ansor_task: ansor.SearchTask


class SymTaskAndInstances(NamedTuple):
    task: "SymTask"
    instances: List[TaskInstance]


class SymTask:
    def __init__(self, sym_dag: SymbolicDAG, sym_ansor_dag: ansor.ComputeDAG):
        self.sym_dag = sym_dag
        self.ansor_task, ansor_policy = self.make_task_from_dag(sym_ansor_dag)
        sketches = ffi.generate_all_sym_sketches(ansor_policy)
        self.sketches = [Sketch(self, sketch) for sketch in sketches]
        self._backbone_to_sketch: Dict[tuple, List[Sketch]] = defaultdict(list)
        for sketch in self.sketches:
            self._backbone_to_sketch[sketch.backbone].append(sketch)

    @property
    def flops_expr(self):
        return self.ansor_task.compute_dag.flop_ct_expr

    def get_flops(self, sizes):
        return int(ffi.subst_by_name(self.flops_expr, sizes))

    def find_sketch(self, backbone: tuple) -> Optional[Sketch]:
        sketches = self._backbone_to_sketch.get(backbone)
        if sketches is None:
            _logger.warning(f"Backbone {backbone} not found")
            return None
        if len(sketches) > 1:
            _logger.warning(f"Backbone {backbone} has multiple sketches")
            return None
        return sketches[0]

    def make_concrete_task(self, size_info: Dict[str, int]):
        ansor_dag = self.sym_dag.make_ansor_compute_dag(size_info)
        ansor_task, policy = self.make_task_from_dag(ansor_dag)
        return ansor_task, policy

    def dag_nodes(self) -> List[RelayOpBuilder]:
        return self.sym_dag.sorted_nodes

    def __str__(self) -> str:
        return str(self.sym_dag)

    def __repr__(self) -> str:
        return f"RelayOperator({repr(self.sym_dag)}, n_sketches={len(self.sketches)})"

    @classmethod
    def make_task_from_dag(
        cls, dag: ansor.ComputeDAG
    ) -> Tuple[ansor.SearchTask, ansor.SketchPolicy]:
        workload_key = register_workload_tensors(dag.workload_key(), dag.tensors)
        task = ansor.SearchTask(
            compute_dag=dag,
            workload_key=workload_key,
            target=utils.TARGET,
            hardware_params=utils.HW_PARAMS,
        )
        policies = ansor.task_scheduler.make_search_policies("default", {}, [task], 1, 1)
        assert len(policies) == 1
        policy = policies[0]
        policy.set_verbose(0)
        return task, policy


def batch_create_tasks(
    tasks: List[utils.AnsorTaskWeight],
    hash_match: bool = True,
    print_tasks: bool = True,
    progress: bool = True,
):
    n_total, n_no_ansor_dag, n_no_nx_dag, n_no_sketches = len(tasks), 0, 0, 0
    grouped: Dict[SymbolicDAG, List[TaskInstance]] = defaultdict(list)
    tasks_ = tqdm(tasks) if progress else tasks
    for i, (task, weight) in enumerate(tasks_):
        concrete_dag = task.compute_dag
        if (dag_result := SymbolicDAG.from_ansor(concrete_dag, hash_match)) is None:
            _logger.debug("Could not convert %s to Symbolic equivalents", concrete_dag)
            n_no_nx_dag += 1
            continue
        sym_dag, size_dict = dag_result
        grouped[sym_dag].append(TaskInstance(i, weight, size_dict, task))
    _logger.info(f"Found {len(grouped)} tasks from {n_total} concrete tasks")
    ret_groups: List[SymTaskAndInstances] = []
    log_print = []
    grouped_ = tqdm(grouped.items(), total=len(grouped)) if progress else grouped.items()
    for sym_dag, instances in grouped_:
        indices = [instance.idx for instance in instances]
        # Run this to make sure all instances have the same size parameters
        size_params, _ = utils.coalesce_dicts([instance.sizes for instance in instances])
        _logger.debug(f"Creating DAG {repr(sym_dag)} (task indices {indices})")
        ansor_dag = sym_dag.make_ansor_compute_dag(None)
        if ansor_dag is None:
            n_no_ansor_dag += len(instances)
            _logger.debug("Skipping task as its DAG generation is not supported")
            continue
        task = SymTask(sym_dag, ansor_dag)
        if len(task.sketches) == 0:
            n_no_sketches += len(instances)
            _logger.debug("Skipping task as no sketch is found")
            continue
        ret_groups.append(SymTaskAndInstances(task, instances))
        if print_tasks:
            log_print.append(f"Task {task}: ")
            log_print.append(f"  Workload key: {task.ansor_task.workload_key}")
            sketches = [
                f"\n    Sketch #{len(sk.backbone)} -> {sk.save_path()}" for sk in task.sketches
            ]
            log_print.append(f"  Sketches:{''.join(sketches)}")
            log_print.append(f"  Size params: {size_params}")
            for index, weight, size_dict, _ in instances:
                flops = task.get_flops(size_dict)
                log_print.append(
                    f"  task {index}; {flops / 1e6:.4f} * {weight} MFLOPs\n"
                    f"    (when sizes = {size_dict})"
                )
            log_print.append("")
    n_success = n_total - n_no_nx_dag - n_no_ansor_dag - n_no_sketches
    _logger.info("%d / %d tasks worked till feature extraction", n_success, n_total)
    _logger.info("%d / %d tasks cannot convert to nx dag", n_no_nx_dag, n_total)
    _logger.info("%d / %d tasks cannot convert back to ansor dag", n_no_ansor_dag, n_total)
    _logger.info("%d / %d tasks does not have sketches", n_no_sketches, n_total)
    if print_tasks:
        _logger.info("Task breakdown:")
        _logger.info("\n".join(log_print))
    return ret_groups


def extract_tasks_(
    model: Union[nn.Module, onnx.ModelProto],  # type: ignore
    example_inputs: Optional[utils.InputSpec] = None,
    save_to: Optional[utils.PathLike] = None,
    entry_pt_name: Optional[str] = None,
):
    tasks = batch_create_tasks(utils.extract_ansor_tasks(model, example_inputs, entry_pt_name))
    if save_to is not None:
        with open(save_to, "wb") as f:
            pkl.dump(tasks, f)
    return tasks


def extract_tasks(model, example_inputs=None, save_to=None, entry_pt_name=None):
    tasks = extract_tasks_(model, example_inputs, save_to, entry_pt_name)
    return [(sym_task, instance) for sym_task, instances in tasks for instance in instances]


def load_and_register_tasks_(task_pkl: utils.PathLike):
    with open(task_pkl, "rb") as f:
        tasks = pkl.load(f)
    assert (
        isinstance(tasks, list)
        and len(tasks) > 0
        and all(isinstance(x, SymTaskAndInstances) for x in tasks)
    )
    tasks = cast(List[SymTaskAndInstances], tasks)
    for _, instances in tasks:
        for inst in instances:
            ansor_task = inst.ansor_task
            ansor.workload_registry.register_workload_tensors(
                ansor_task.workload_key, ansor_task.compute_dag.tensors
            )
    return tasks


def load_and_register_tasks(task_pkl: utils.PathLike):
    tasks = load_and_register_tasks_(task_pkl)
    return [(sym_task, instance) for sym_task, instances in tasks for instance in instances]


def extract_tenset_pickle_tasks(tenset_task_pkl: PathLike, felix_task_pkl: PathLike):
    def check_pair(pair) -> AnsorTaskWeight:
        assert isinstance(pair[0], ansor.SearchTask)
        assert isinstance(pair[1], int)
        return pair

    def patch_one_instance(sym_task, inst):
        ansor_task = inst.ansor_task
        new_ansor_dag = sym_task.sym_dag.make_ansor_compute_dag(inst.sizes)
        if new_ansor_dag is None:
            return None
        if ansor_task.compute_dag.flop_ct == -1:
            # Must do this to make sure the flops are updated in the task
            # (otherwise copy on write and nothing happens)
            dag = ansor_task.compute_dag
            dag.flop_ct = new_ansor_dag.flop_ct
            ansor_task.compute_dag = dag
        return TaskInstance(inst.idx, inst.weight, inst.sizes, ansor_task)

    def patch_instances(sym_task, instances):
        return [
            inst_ for inst in instances if (inst_ := patch_one_instance(sym_task, inst)) is not None
        ]

    if Path(felix_task_pkl).is_file():
        return load_and_register_tasks(felix_task_pkl)
    with open(tenset_task_pkl, "rb") as fr:
        tasks = pkl.load(fr)
    assert isinstance(tasks, list) and len(tasks) > 0
    tasks = [check_pair(task) for task in tasks]
    # Register workloads & override hardware params
    for i in range(len(tasks)):
        task, weight = tasks[i]
        ansor.workload_registry.register_workload_tensors(
            task.workload_key, task.compute_dag.tensors
        )
        task = ansor.SearchTask(
            workload_key=task.workload_key, target=TARGET, hardware_params=HW_PARAMS
        )
        tasks[i] = task, weight
    # Don't check the hash value of generated tasks against Tenset's.
    # This forfeits a key safety check but is necessary as Tenset's DAGs
    # are generated with TVM of an older version.
    to_save = [
        SymTaskAndInstances(sym_task, patch_instances(sym_task, instances))
        for sym_task, instances in batch_create_tasks(tasks, hash_match=False, print_tasks=False)
    ]
    with open(felix_task_pkl, "wb") as fw:
        pkl.dump(to_save, fw)
    return [(sym_task, instance) for sym_task, instances in to_save for instance in instances]


@dataclass
@total_ordering
class LatAndThruput:
    latency_us: float
    thruput_tflops: float

    @classmethod
    def from_thruput(cls, flops: float, thruput_tflops: float):
        thruput_flops = thruput_tflops * 1e12
        latency_us = flops / thruput_flops * 1e6
        return cls(latency_us, thruput_tflops)

    @classmethod
    def from_latency_s(cls, flops: float, latency_sec: float):
        latency_us = latency_sec * 1e6
        thruput_tflops = flops / latency_sec / 1e12
        return cls(latency_us, thruput_tflops)

    def __lt__(self, other) -> bool:
        if not isinstance(other, LatAndThruput):
            return NotImplemented
        return self.thruput_tflops < other.thruput_tflops


@dataclass
@total_ordering
class PerfScore:
    score: float

    def __lt__(self, other) -> bool:
        if not isinstance(other, PerfScore):
            return NotImplemented
        return self.score < other.score


Performance = Union[LatAndThruput, PerfScore]


def print_perf(shape: Optional[Performance], weight: int):
    if isinstance(shape, LatAndThruput):
        return f"{shape.latency_us:.2f} us (*{weight}), {shape.thruput_tflops:.4f} TFLOPs"
    if isinstance(shape, PerfScore):
        return f"score {shape.score:.3f} (weight={weight})"
    if shape is None:
        return "N/A"
    assert False


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


class TaskPerfFunc(nn.Module):
    def __init__(self, task: SymTask, sizes: Dict[str, int], perf_model: MLPCostModel) -> None:
        super().__init__()
        self.sym_task = task
        self.sizes = sizes
        self.ansor_task, self.ansor_policy = task.make_concrete_task(sizes)
        sketches = [SketchPerfFunc(self, sketch, perf_model) for sketch in task.sketches]
        self._sketches = nn.ModuleList(sketches)
        self.is_throughput = perf_model.is_throughput
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
        perfs, constraints = utils.transpose2(
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
                perf = max_perfs[conf_i].item()
                if self.is_throughput:
                    perf = LatAndThruput.from_thruput(self.flops, perf)
                else:
                    perf = PerfScore(perf)
                ret.append(ConfigInfo(config_dict, sketch, perf))
        _logger.info(
            f"get_best_configs: n_invalid={n_invalid}, n_dup={n_dup}, n_configs={len(ret)}"
        )
        return sorted(ret, key=lambda x: x.pred_perf, reverse=True)

    def measure_configs_latency_(self, configs: List[ConfigInfo]):
        from tvm.auto_scheduler.measure import ProgramMeasurer

        measurer = ProgramMeasurer(ansor.LocalBuilder(), utils.MEASURER, [], False)
        results = ffi.measure_mis_performance(
            self.ansor_policy, measurer, [c.get_measure_input() for c in configs]
        )
        for c, result in zip(configs, results):
            if result.error_no != 0:
                _logger.warning(result.error_msg)
            c.measure_result = result
        return configs

    def __repr__(self) -> str:
        return (
            f"TaskLatFunc({self.sym_task}, sizes={self.sizes}," f"sketches={len(self._sketches)})"
        )

    __str__ = __repr__
