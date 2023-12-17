import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm
from tvm import auto_scheduler as ansor
from tvm import felix
from tvm.auto_scheduler.measure import (
    BuildResult,
    LocalBuilder,
    LocalRunner,
    MeasureInput,
)
from tvm.auto_scheduler.measure_record import save_records

TARGET = str(felix.utils.TARGET.model)
TENSET_TASKS_PKL = Path("lightning_logs/tenset/network_info/all_tasks.pkl")
TO_MEASURE_PROGRAM_FOLDER = Path("lightning_logs/tenset/to_measure_programs")
MEASURE_RECORD_FOLDER = Path(f"lightning_logs/{TARGET}/measure_records")
RUNNER_KWARGS = {
    "timeout": 15,
    "min_repeat_ms": 30,
    "repeat": 3,
    "enable_cpu_cache_flush": False,
}
BUILDER = LocalBuilder()
MINI_BATCH_SIZE = 32
MAX_QSIZE = 1024

logger = logging.getLogger(__name__)


@dataclass
class TaskRecord:
    sym_task: felix.SymTask
    instance: felix.TaskInstance
    tenset_src: Path
    output_path: Path

    @property
    def ansor_task(self):
        return self.instance.ansor_task


def enumerate_tasks(groups: List[str], n_tasks_per_group: Optional[int]) -> List[TaskRecord]:
    groups_ = set(groups)

    def match_in_filter(sym_task):
        if not filter:
            return ""
        for n in sym_task.sym_dag.sorted_nodes:
            if str(n) in groups_:
                return str(n)
        return None

    def clean_name(x):
        return str(x).replace(" ", "").replace('"', "").replace("'", "") + ".json"

    tasks_info = felix.extract_tenset_pickle_tasks(
        TENSET_TASKS_PKL, MEASURE_RECORD_FOLDER / "felix_tasks.pkl"
    )
    tasks = defaultdict(list)
    for sym_task, inst in tasks_info:
        if (group := match_in_filter(sym_task)) is None:
            continue
        task = inst.ansor_task
        filename = clean_name((task.workload_key, str(task.target.kind)))
        ifile = TO_MEASURE_PROGRAM_FOLDER / filename
        opath = Path(group) / filename  # When group == "" this'll just be Path(filename)
        tasks[group].append(TaskRecord(sym_task, inst, ifile, opath))
    ret_tasks = []
    for group, grp_tasks in tasks.items():
        random.shuffle(grp_tasks)
        if n_tasks_per_group is not None:
            ret_tasks.extend((selected := grp_tasks[:n_tasks_per_group]))
            logger.info("Selected %d / %d tasks for %s", len(selected), len(grp_tasks), group)
        else:
            ret_tasks.extend(grp_tasks)
            logger.info("Loaded %d tasks for %s", len(grp_tasks), group)
    return ret_tasks


def _measure_configs(queue: Queue, runner: LocalRunner):
    mis: List[MeasureInput] = []
    bresults: List[BuildResult] = []
    out_files: List[Path] = []

    def batch_operate():
        assert len(mis) == len(bresults) == len(out_files)
        logger.info("Fetched %d configs to measure from the queue", len(mis))
        logger.info("%d configs in the queue to be measured", queue.qsize())
        mres = runner.run(mis, bresults)
        n_suc = 0
        for inp, res, out_file in zip(mis, mres, out_files):
            if res.error_no != 0:
                logger.debug(f"Error: {res.error_no} {res.error_msg}")
            else:
                n_suc += 1
            save_records(out_file.as_posix(), [inp], [res])
        logger.info("%d out of %d configs measured successfully", n_suc, len(mis))

    while True:
        next_item = queue.get()
        if next_item is None:
            break
        br, mi, out_file = next_item
        mis.append(mi)
        bresults.append(br)
        out_files.append(out_file)
        if len(mis) == MINI_BATCH_SIZE:
            batch_operate()
            mis, bresults, out_files = [], [], []
    if mis:
        batch_operate()


def _produce_configs(queue: Queue, tasks: List[TaskRecord], n_per_task: int, output_prefix: Path):
    for task in tqdm(tasks, leave=None):
        sym_task, inst = task.sym_task, task.instance
        flops = task.ansor_task.compute_dag.flop_ct
        output_file = output_prefix / task.output_path
        # fmt: off
        logger.info(
            "Task %d (%s)\n    FLOPs = %s (ansor), params = %s\n    file = %s -> %s",
            inst.idx, sym_task, flops, inst.sizes, task.tenset_src, output_file
        )
        # fmt: on
        inps, _ = ansor.RecordReader(task.tenset_src.as_posix()).read_lines()
        rand_idx = torch.randperm(len(inps))[:n_per_task]
        # Do this to set the target/target_host settings to ours in the measure input:
        mis = [ansor.MeasureInput(task.ansor_task, inps[i].state) for i in rand_idx.tolist()]
        logger.info("Randomly selected %d configs", len(mis))
        build_results: List[BuildResult] = BUILDER.build(mis)
        assert len(build_results) == len(mis)
        n_build_suc = 0
        for br, mi in zip(build_results, mis):
            if br.error_no != 0:
                logger.debug(f"Build error: {br.error_no} {br.error_msg}")
            else:
                queue.put((br, mi, output_file))  # Wait can happen here
                n_build_suc += 1
        logger.info("%d out of %d configs built successfully", n_build_suc, len(mis))
        logger.info("%d configs in the queue to be measured", queue.qsize())


def seed_all(seed: int):
    import numpy as np

    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)


def main():
    params = {
        "n_tasks": 100,
        "task_groups": ["Conv2d", "Dense", "BatchMatmul", "DepthwiseConv2d"],
        "configs_per_task": 256,
        "devices": [0, 1, 2, 3],
        "config_selector": None,
    }

    seed_all(0)
    MEASURE_RECORD_FOLDER.mkdir(parents=True, exist_ok=True)
    felix.init_logging(MEASURE_RECORD_FOLDER)
    logger.info("Runner using config %s", RUNNER_KWARGS)
    logger.info("Loading all tasks...")
    tasks = enumerate_tasks(params["task_groups"], params["n_tasks"])

    if (config_select := params["config_selector"]) is not None:
        raise NotImplementedError("TODO: Implement config selection")

    n_confs = params["configs_per_task"]
    output_prefix = MEASURE_RECORD_FOLDER / "full"
    task_config_queue = Queue(MAX_QSIZE)
    config_producer = Process(
        target=_produce_configs, args=(task_config_queue, tasks, n_confs, output_prefix)
    )
    config_producer.start()
    config_consumers = []
    for device_idx in params["devices"]:
        runner = ansor.measure.LocalRunner(**RUNNER_KWARGS, device=device_idx)
        config_consumer = Process(target=_measure_configs, args=(task_config_queue, runner))
        config_consumer.start()
        config_consumers.append(config_consumer)
    logger.info("Started %d runners", len(config_consumers))

    config_producer.join()
    logger.info("Config producer finished")
    for config_consumer in config_consumers:
        # Put a None for every consumer to signal that we're done,
        # and wait for them to finish.
        task_config_queue.put(None)
        config_consumer.join()


if __name__ == "__main__":
    main()
