import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from tvm import auto_scheduler as ansor
from tvm import felix
from tvm.auto_scheduler.feature import get_per_store_features_from_states
from tvm.felix import ffi, utils

TARGET = str(utils.TARGET.model)
TENSET_TASKS_PKL = Path("lightning_logs/tenset/network_info/all_tasks.pkl")
TO_MEASURE_PROGRAM_FOLDER = Path("lightning_logs/tenset/to_measure_programs")
MEASURE_RECORD_FOLDER = Path(f"lightning_logs/{TARGET}/measure_records")
RUNNER_KWARGS = {
    "timeout": 15,
    "min_repeat_ms": 100,
    "repeat": 1,
    "enable_cpu_cache_flush": False,
}
BUILDER = ansor.measure.LocalBuilder()

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


def get_confs_to_measure(taskr: TaskRecord, cost_model, n_total, n_best):
    task = taskr.ansor_task
    inps, _ = ansor.RecordReader(taskr.tenset_src.as_posix()).read_lines()
    if cost_model is None:
        assert n_best == 0
        best_idx = torch.tensor([])
        rand_idx = torch.randperm(len(inps))[:n_total]
    else:
        features = get_per_store_features_from_states([inp.state for inp in inps], task)
        features = [torch.tensor(f.astype(np.float32)) for f in features]
        predictions = cost_model.forward_on_batch(features)
        sorted_indices = torch.argsort(predictions, descending=True)
        best_idx, rest = sorted_indices[:n_best], sorted_indices[n_best:]
        n_rand = n_total - n_best
        rand_idx = rest[torch.randperm(len(rest))[:n_rand]]
        pred_usec = 1e6 / torch.exp(predictions[best_idx])
        logger.info(f"{n_best} best configs with predicted costs (usecs)\n  %s", pred_usec)
    # Do this to set the target/target_host settings to ours in the measure input:
    ret = [ansor.MeasureInput(task, inps[i].state) for i in best_idx.tolist() + rand_idx.tolist()]
    logger.info("Got %d configs (%d best and %d random)", len(ret), len(best_idx), len(rand_idx))
    return ret


def measure_configs(inps, runners, output_file):
    from threading import Thread

    def measure_configs(runner, inps_):
        measurer = ansor.measure.ProgramMeasurer(
            BUILDER,
            runner,
            [ansor.RecordToFile(str(output_file))],
            verbose=0,
            max_continuous_error=16,
        )
        ress = ffi.measure_performance(measurer, inps_.tolist())
        for inp, res in zip(inps_, ress):
            if res.error_no != 0:
                logger.info(f"Error: {res.error_no} ...{res.error_msg[-64:]}")
                continue
            inp_res_pairs.append((inp, res))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    inp_res_pairs = []
    chunks = np.array_split(np.array(inps), len(runners))
    threads = [
        Thread(target=measure_configs, args=(runner, inps_))
        for inps_, runner in zip(chunks, runners)
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    logger.info(f"Measurement finished for task; {len(inp_res_pairs)} succeeded out of {len(inps)}")
    return inp_res_pairs


def measure_on_all_tasks(tasks, dbuilder, runners, output_prefix, n_total, n_best, cost_model=None):
    if cost_model is None:
        assert n_best == 0
    for task in tqdm(tasks, leave=None):
        sym_task, inst = task.sym_task, task.instance
        flops = task.ansor_task.compute_dag.flop_ct
        # fmt: off
        logger.info(
            "Task %d (%s)\n    FLOPs = %s (ansor), params = %s",
            inst.idx, sym_task, flops, inst.sizes
        )
        # fmt: on
        to_measure = get_confs_to_measure(task, cost_model, n_total, n_best)
        inp_res_pairs = measure_configs(to_measure, runners, output_prefix / task.output_path)
        costs = np.array([[float(t) * 1e6 for t in res.costs] for _, res in inp_res_pairs])
        costs = costs.mean(axis=1)
        logger.info(
            "Config costs (us):\n  Best configs: %s\n  Rand configs: %s",
            costs[:n_best],
            costs[n_best:],
        )
        felix.add_to_dataset_builder(dbuilder, sym_task, inst, inp_res_pairs, True)
        dataset = dbuilder.to_dataset()
        if cost_model is not None:
            if len(dataset) < 512:
                logger.info("Not enough data to train cost model")
            else:
                cost_model.train_self(dataset, 512, 50, 5, 1e-4)
    torch.save((dataset := dbuilder.to_dataset()), output_prefix / "ansor_dataset.pkl")
    return dataset


def seed_all(seed: int):
    import numpy as np

    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)


def main():
    params = {
        "n_tasks": 100,
        "task_groups": ["Conv2d", "Dense", "DepthwiseConv2d"],
        "configs_per_round": 64,
        "best_configs": [56] * 2,
        "devices": [0, 1],
        "initial_dataset": None,
        "initial_model": "lightning_logs/a5000/measure_records/bootstrap/epoch=138-loss=0.3916.ckpt",
    }

    seed_all(0)
    MEASURE_RECORD_FOLDER.mkdir(parents=True, exist_ok=True)
    felix.init_logging(MEASURE_RECORD_FOLDER)
    logger.info("Runner using config %s", RUNNER_KWARGS)
    logger.info("Loading all tasks...")
    tasks = enumerate_tasks(params["task_groups"], params["n_tasks"])

    runners = [ansor.measure.LocalRunner(**RUNNER_KWARGS, device=idx) for idx in params["devices"]]
    n_confs, dataset_path = params["configs_per_round"], params["initial_dataset"]
    dbuilder = felix.DatasetBuilder()
    if dataset_path is not None:
        dataset = torch.load(dataset_path)
        assert isinstance(dataset, ansor.cost_model.SegmentDataset)
        dbuilder.add_dataset(dataset)
    if (model_ckpt := params["initial_model"]) is not None:
        cost_model = felix.MLPModelPLWrapper.load_from_checkpoint(model_ckpt)
    else:
        if dataset_path is None:
            dataset = measure_on_all_tasks(
                tasks, dbuilder, runners, MEASURE_RECORD_FOLDER / "bootstrap", n_confs, 0
            )
        cost_model = felix.MLPModelPLWrapper(n_features=164)
        cost_model.train_self(dbuilder.to_dataset(), 512, 200, 10, 1e-4)
    for n_best in tqdm(params["best_configs"], leave=None):
        prefix = MEASURE_RECORD_FOLDER / "full"
        measure_on_all_tasks(tasks, dbuilder, runners, prefix, n_confs, n_best, cost_model)


if __name__ == "__main__":
    main()
