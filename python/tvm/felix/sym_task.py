import logging
import pickle as pkl
from collections import defaultdict
from typing import Dict, List, NamedTuple, Optional, Tuple, Union, cast

import onnx
from torch import nn
from tqdm import tqdm
from tvm import auto_scheduler as ansor
from tvm.auto_scheduler.workload_registry import register_workload_tensors

from . import ffi, utils
from .sketch import Sketch
from .sym_dag import RelayOpBuilder, SymbolicDAG

_logger = logging.getLogger(__name__)
__all__ = [
    "SymTask",
    "SymTaskAndInstances",
    "batch_create_tasks",
    "extract_tasks",
    "extract_tasks_",
    "load_and_register_tasks",
    "load_and_register_tasks_",
]
AnsorTaskWeight = Tuple[ansor.SearchTask, int]


class TaskInstance(NamedTuple):
    idx: int
    weight: int
    sizes: dict
    ansor_task: ansor.SearchTask


class SymTaskAndInstances(NamedTuple):
    task: "SymTask"
    instances: List[TaskInstance]


class SymTask:
    def __init__(
        self,
        sym_dag: SymbolicDAG,
        sym_ansor_dag: ansor.ComputeDAG,
        const_names: List[str],
    ):
        self.sym_dag = sym_dag
        self.ansor_dag = sym_ansor_dag
        self.const_names = const_names
        self.ansor_task, self.ansor_policy = self.make_task_from_dag(sym_ansor_dag)
        sketches = ffi.generate_all_sym_sketches(self.ansor_policy)
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
    tasks: List[AnsorTaskWeight],
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
        size_dicts = [instance.sizes for instance in instances]
        size_params, _ = utils.coalesce_dicts(size_dicts)
        _logger.debug(f"Creating DAG {repr(sym_dag)} (task indices {indices})")
        ansor_dag = sym_dag.make_ansor_compute_dag(None)
        if ansor_dag is None:
            n_no_ansor_dag += len(instances)
            _logger.debug("Skipping task as its DAG generation is not supported")
            continue
        task = SymTask(sym_dag, ansor_dag, size_params)
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


def extract_tasks(
    model: Union[nn.Module, onnx.ModelProto],  # type: ignore
    example_inputs: Optional[utils.InputSpec] = None,
    save_to: Optional[utils.PathLike] = None,
    entry_pt_name: Optional[str] = None,
):
    logging.getLogger("topi").setLevel(logging.WARNING)
    mod, params = utils.get_tvm_mod(model, example_inputs, entry_pt_name=entry_pt_name)
    _logger.debug("Module (Relay IR): \n%s", mod["main"])
    tasks, task_weights = ansor.extract_tasks(
        mod["main"], params, utils.TARGET, hardware_params=utils.HW_PARAMS
    )
    assert len(tasks) == len(task_weights)
    task_weights = [int(w) for w in task_weights]
    tasks_weights = sorted(zip(tasks, task_weights), key=lambda x: utils.parse_task(x[0])[0][0])
    tasks = batch_create_tasks(tasks_weights)
    if save_to is not None:
        with open(save_to, "wb") as f:
            pkl.dump(tasks, f)
    return tasks


def extract_tasks_(model, example_inputs=None, save_to=None, entry_pt_name=None):
    tasks = extract_tasks(model, example_inputs, save_to, entry_pt_name)
    return [(sym_task, instance) for sym_task, instances in tasks for instance in instances]


def load_and_register_tasks(task_pkl: utils.PathLike):
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


def load_and_register_tasks_(task_pkl: utils.PathLike):
    tasks = load_and_register_tasks(task_pkl)
    return [(sym_task, instance) for sym_task, instances in tasks for instance in instances]
