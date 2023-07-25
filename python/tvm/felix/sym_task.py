import logging
import pickle as pkl
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union, cast

import onnx
from torch import nn
from tqdm import tqdm
from tvm import auto_scheduler as ansor
from tvm.auto_scheduler.loop_state import StateObject
from tvm.auto_scheduler.workload_registry import register_workload_tensors

from . import ffi, utils
from .features import TorchFeatures
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

FEATURE_CACHE_PATH = Path(Path("~").expanduser(), ".tvm", "felix", "features")


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

    def find_sketch(self, backbone: tuple) -> Optional["Sketch"]:
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


class Sketch:
    def __init__(self, task, sym_state: StateObject):
        self.parent_task: SymTask = task
        self.state_repr = str(sym_state)
        self.tr_steps = sym_state.transform_steps
        self.code, self.context = ffi.generate_code_for_state(task.ansor_task, sym_state, True)
        self.backbone = ffi.extract_backbone(self.tr_steps)

        _logger.debug("Sketch transformation steps: %s", ffi.print_state_tr_steps(sym_state))
        _logger.debug("Code: %s", self.code)
        _logger.debug("With loop sizes:\n%s", self.context.to_varmap())

    def state_hash(self) -> str:
        import hashlib

        md5 = hashlib.md5()
        md5.update(self.state_repr.encode())
        return md5.hexdigest()

    def save_path(self) -> Path:
        return FEATURE_CACHE_PATH / f"{self.state_hash()}.json"

    def fetch_features(
        self,
        sizes: Dict[str, int],
        prime_factorize: bool = True,
        max_n_buf: int = 5,
        cache_line_size: int = 64,
    ):
        path = self.save_path()
        path.parent.mkdir(exist_ok=True, parents=True)
        features = ffi.get_feature_pack(
            self.code,
            self.context,
            HW_PARAMS,
            sizes,
            cache_line_size,
            max_n_buf,
            prime_factorize,
            path.with_suffix("").as_posix(),
        )
        return TorchFeatures.from_feat_pack(features)

    def __str__(self) -> str:
        return f"Sketch({self.backbone} from {self.parent_task})"

    __repr__ = __str__
