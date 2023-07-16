import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import tvm
from tqdm import tqdm, trange
from tvm import auto_scheduler as ansor
from tvm import felix
from tvm.felix import ffi
from tvm.felix.utils import TARGET

NETWORK_INFO_FOLDER = Path("lightning_logs/tenset/network_info")
TO_MEASURE_PROGRAM_FOLDER = Path("lightning_logs/tenset/to_measure_programs")
MEASURE_RECORD_FOLDER = Path(f"lightning_logs/{TARGET.model}/measure_records")
MEASURE_BATCH_SIZE = 32
RUNNER_KWARGS = {
    "timeout": 15,
    "number": 1,
    "min_repeat_ms": 0,
    "enable_cpu_cache_flush": False,
}

logger = logging.getLogger(__name__)


@dataclass
class TaskRecord:
    sym_task: felix.SymTask
    instance: felix.TaskInstance
    input_file: Path
    output_file: Path

    @property
    def ansor_task(self):
        return self.instance.ansor_task

    def n_repeats(self):
        flops = self.instance.ansor_task.compute_dag.flop_ct
        if flops >= 2416443392.0:
            return 4
        elif flops >= 834928640.0:
            return 6
        elif flops >= 2097152.0:
            return 8
        else:
            return 10


def enumerate_tasks(filter: List[str]) -> List[TaskRecord]:
    filter_ = set(filter)

    def match_in_filter(sym_task):
        if not filter:
            return MEASURE_RECORD_FOLDER
        for n in sym_task.sym_dag.sorted_nodes:
            if str(n) in filter_:
                return MEASURE_RECORD_FOLDER / str(n)
        return None

    def clean_name(x):
        return str(x).replace(" ", "").replace('"', "").replace("'", "") + ".json"

    tasks_info = felix.extract_tenset_pickle_tasks(
        NETWORK_INFO_FOLDER / "all_tasks.pkl", MEASURE_RECORD_FOLDER / "felix_tasks.pkl"
    )
    ret = []
    for sym_task, inst in tasks_info:
        if (output_dir := match_in_filter(sym_task)) is None:
            continue
        task = inst.ansor_task
        filename = clean_name((task.workload_key, str(task.target.kind)))
        ifile, ofile = TO_MEASURE_PROGRAM_FOLDER / filename, output_dir / filename
        ret.append(TaskRecord(sym_task, inst, ifile, ofile))
    return ret


def prefilter_task_configs(
    taskr: TaskRecord,
    measure_inputs: List[ansor.MeasureInput],
    cost_model: felix.MLPModelPLWrapper,
    n_best: int,
    n_rand: int,
):
    import numpy as np
    from tvm.auto_scheduler.feature import get_per_store_features_from_states

    states = [inp.state for inp in measure_inputs]
    try:
        features = get_per_store_features_from_states(states, taskr.ansor_task)
    except tvm.TVMError as e:
        # Ansor feature extraction failed, skip this task
        logger.warning("Failed to extract features: %s", e)
        return []
    features = [torch.tensor(f.astype(np.float32)) for f in features]
    predictions = cost_model.forward_on_batch(features)
    sorted_indices = torch.argsort(predictions, descending=True)
    best_idx, rest = sorted_indices[:n_best], sorted_indices[n_best:]
    rand_idx = rest[torch.randperm(len(rest))[:n_rand]]
    logger.info(
        "Taking %d random configs and %d best configs with predicted costs\n  %s",
        n_rand,
        n_best,
        predictions[best_idx],
    )
    return [measure_inputs[i] for i in best_idx.tolist() + rand_idx.tolist()]


def make_measurer(verbose, log_filename, repeat: int):
    builder = ansor.measure.LocalBuilder()
    runner = ansor.measure.LocalRunner(**RUNNER_KWARGS, repeat=repeat)
    measurer = ansor.measure.ProgramMeasurer(
        builder,
        runner,
        [ansor.RecordToFile(str(log_filename))],
        verbose=verbose,
    )
    return measurer


def remeasure_file(taskr: TaskRecord, skip_existing: bool, prefilter=None):
    if taskr.output_file.is_file() and skip_existing:
        logger.info("File %s already exists, skipping", taskr.output_file)
        return
    logger.info("Will write to file %s", taskr.output_file)
    measurer = make_measurer(0, taskr.output_file, taskr.n_repeats())
    # Read reference measurement inputs
    inputs, _ = ansor.RecordReader(taskr.input_file.as_posix()).read_lines()
    inputs = list(inputs)
    if prefilter:
        inputs_ = prefilter(taskr, inputs)
        logger.info("Prefiltering kept %d of %d configs", len(inputs_), len(inputs))
        inputs = inputs_
    if len(inputs) == 0:
        logger.info("No configs left, skipping.")
        return
    empty_policy = ansor.search_policy.EmptyPolicy(taskr.ansor_task)
    # Do measurement
    logger.info("Started measuring %d configs", len(inputs))
    bs = MEASURE_BATCH_SIZE
    for from_ in trange(0, len(inputs), bs):
        to = from_ + bs
        states = [inp.state for inp in inputs[from_:to]]
        to = min(to, len(inputs))
        results = ffi.measure_state_performance(empty_policy, measurer, states)
        n_failed = 0
        for result in results:
            if result.error_no != 0:
                n_failed += 1
                tqdm.write(f"Error: {result.error_no} {result.error_msg}")
                continue
            flops = taskr.ansor_task.compute_dag.flop_ct
            gflops_sec = [flops / 1e9 / t for t in result.costs]
            tqdm.write(f"{gflops_sec}")
        logger.info(f"{n_failed} failed out of {len(states)} in batch {from_ // bs}")
    logger.info("Measurement finished for task.")


def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--n-tasks", type=int)
    parser.add_argument("--n-configs", type=int)
    parser.add_argument("--task-filter", type=str, nargs="*")
    parser.add_argument("--cost-model", type=Path)
    parser.add_argument("--gpu-index", type=int, default=0)
    return parser.parse_args()


def seed_all(seed: int):
    import numpy as np

    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)


if __name__ == "__main__":
    args = parse_args()
    MEASURE_RECORD_FOLDER.mkdir(parents=True, exist_ok=True)
    felix.init_logging(MEASURE_RECORD_FOLDER)
    logger.info("Batch size of measurement = %d", MEASURE_BATCH_SIZE)
    RUNNER_KWARGS["device"] = args.gpu_index
    logger.info("Runner using config %s", RUNNER_KWARGS)
    logger.info("Loading all tasks...")
    tasks = enumerate_tasks(args.task_filter)
    seed_all(0)
    random.shuffle(tasks)
    if args.n_tasks is not None:
        logger.info("Loaded first %d from %d tasks", args.n_tasks, len(tasks))
        tasks = tasks[: args.n_tasks]
    else:
        logger.info("Loaded %d tasks", len(tasks))
    if args.cost_model is not None:
        cost_model = felix.MLPModelPLWrapper.load_from_checkpoint(args.cost_model)
        if args.n_configs is None:
            raise ValueError("Must specify --n-configs when using a cost model")
        n_best = args.n_configs // 2
        n_rand = args.n_configs - n_best
        conf_filter = lambda task, inputs: prefilter_task_configs(
            task, inputs, cost_model, n_best, n_rand
        )
    elif args.n_configs is not None:
        conf_filter = lambda _, inputs: inputs[-args.n_configs :]
    else:
        conf_filter = None
    for task in tqdm(tasks, leave=None):
        ansor_dag = task.ansor_task.compute_dag
        our_ansor_dag = task.sym_task.ansor_task.compute_dag
        # fmt: off
        logger.info(
            "Task %d:\n  Original DAG: %s\n  Parsed DAG: %s\n    with params = %s\n  FLOPs: %s (ansor)",
            task.instance.idx, ansor_dag, our_ansor_dag, task.instance.sizes, ansor_dag.flop_ct
        )
        # fmt: on
        remeasure_file(task, True, conf_filter)
