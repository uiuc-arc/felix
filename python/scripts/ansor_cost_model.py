import argparse
import logging
import pickle as pkl
import random
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tvm import auto_scheduler as ansor
from tvm import felix
from tvm.auto_scheduler.cost_model import (
    XGBModel,
    make_dataset_from_log_file,
    metric_pairwise_cmp_acc,
    metric_peak_score,
    metric_r_squared,
    metric_rmse,
)
from tvm.felix import utils

ALL_TASKS_PKL = "lightning_logs/tenset/network_info/all_tasks.pkl"
ALL_TASKS_PKL_OUT = "lightning_logs/tenset/network_info/all_tasks.pkl"
logger = logging.getLogger(__name__)


def parse_args():
    def varargs(args):
        d = vars(args)
        d.pop("func")
        d.pop("command")
        return d

    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    dataset = commands.add_parser("dataset")
    dataset.add_argument("--logs", required=True, nargs="+", type=str)
    dataset.add_argument("--out-file", type=Path, default=Path("dataset.pkl"))
    dataset.add_argument("--seed", type=int, default=0)
    dataset.add_argument("--sample-in-files", type=int)
    dataset.add_argument("--min-sample-size", type=int, default=48)
    dataset.set_defaults(func=lambda args: make_dataset(**varargs(args)))

    train = commands.add_parser("train")
    train.add_argument("--dataset", nargs="+", required=True, type=Path, dest="datasets")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument(
        "--split-scheme",
        type=str,
        choices=["by_task", "within_task", "by_target"],
        default="within_task",
    )
    train.add_argument("--train-ratio", type=float, default=0.9)
    train.add_argument("--out-file", type=Path, default=Path("xgb.pkl"))
    train.set_defaults(func=lambda args: train_zero_shot(**varargs(args)))

    return parser.parse_args()


def load_and_register_tasks(
    task_pkl: utils.PathLike, override_hw: bool = False
) -> List[utils.AnsorTaskWeight]:
    def check_pair(pair) -> Tuple[ansor.SearchTask, int]:
        assert isinstance(pair[0], ansor.SearchTask)
        assert isinstance(pair[1], int)
        return pair

    with open(task_pkl, "rb") as f:
        tasks = pkl.load(f)
    assert isinstance(tasks, list) and len(tasks) > 0
    tasks = [check_pair(task) for task in tasks]
    for i in range(len(tasks)):
        task, weight = tasks[i]
        ansor.workload_registry.register_workload_tensors(
            task.workload_key, task.compute_dag.tensors
        )
        if override_hw:
            task = ansor.SearchTask(
                workload_key=task.workload_key, target=utils.TARGET, hardware_params=utils.HW_PARAMS
            )
            tasks[i] = task, weight
    return tasks


def seed_all(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)


def evaluate_model(model: XGBModel, valid_set):
    # make prediction
    prediction = model.predict_on_dataset(valid_set)

    # compute weighted average of metrics over all tasks
    tasks = list(valid_set.tasks())
    weights = [len(valid_set.throughputs[t]) for t in tasks]
    logger.info("Test set sizes: %s", weights)

    rmse_list = []
    r_sqaured_list = []
    pair_acc_list = []
    peak_score1_list = []
    peak_score5_list = []
    plot_points = []
    for task in tasks:
        preds = prediction[task]
        labels = valid_set.throughputs[task]
        rmse_list.append(np.square(metric_rmse(preds, labels)))
        r_sqaured_list.append(metric_r_squared(preds, labels))
        pair_acc_list.append(metric_pairwise_cmp_acc(preds, labels))
        peak_score1_list.append(metric_peak_score(preds, labels, 1))
        peak_score5_list.append(metric_peak_score(preds, labels, 5))
        plot_points.extend(list(zip(preds, labels)))

    rmse = np.sqrt(np.average(rmse_list, weights=weights))
    r_sqaured = np.average(r_sqaured_list, weights=weights)
    pair_acc = np.average(pair_acc_list, weights=weights)
    peak_score1 = np.average(peak_score1_list, weights=weights)
    peak_score5 = np.average(peak_score5_list, weights=weights)
    eval_res = {
        "RMSE": rmse,
        "R^2": r_sqaured,
        "pairwise comparision accuracy": pair_acc,
        "average peak score@1": peak_score1,
        "average peak score@5": peak_score5,
    }
    return eval_res, plot_points


def plot_and_save(data_points: list, out_file: Path):
    fig, ax = plt.subplots(1, 1)
    xs, ys = zip(*data_points)
    ax.scatter(xs, ys, s=1)
    ax.set_xlabel("Predicted throughput score")
    ax.set_ylabel("Measured throughput score")
    fig.savefig(out_file.as_posix(), dpi=300)
    plt.close(fig)


def make_dataset(
    logs: List[Path],
    out_file: Path,
    sample_in_files: int,
    min_sample_size: int,
    seed: int,
):
    seed_all(seed)
    logger.info("Load tasks...")
    load_and_register_tasks(ALL_TASKS_PKL)
    if sample_in_files:
        logs = random.sample(logs, sample_in_files)
    logger.info("Featurize measurement records...")
    make_dataset_from_log_file(logs, out_file, min_sample_size)


def train_zero_shot(
    datasets: List[Path],
    out_file: Path,
    train_ratio: float,
    split_scheme: str,
    seed: int,
):
    seed_all(seed)
    logger.info("Load tasks...")
    load_and_register_tasks(ALL_TASKS_PKL)
    with open(datasets[0], "rb") as f:
        dataset = pkl.load(f)
    for dataset_path in datasets[1:]:
        with open(dataset_path, "rb") as f:
            dataset.update_from_dataset(pkl.load(f))
    # Split dataset
    if split_scheme == "within_task":
        train_set, valid_set = dataset.random_split_within_task(train_ratio)
    elif split_scheme == "by_task":
        train_set, valid_set = dataset.random_split_by_task(train_ratio)
    elif split_scheme == "by_target":
        train_set, valid_set = dataset.random_split_by_target(train_ratio)
    else:
        raise ValueError("Invalid split scheme: " + split_scheme)

    logger.info("Train set: %d. Task 0 = %s", len(train_set), train_set.tasks()[0])
    if len(valid_set) == 0:
        valid_set = train_set
    logger.info("Test set:  %d. Task 0 = %s", len(valid_set), valid_set.tasks()[0])
    # Make models
    model = XGBModel()
    model.plan_size = 1
    # Train the model
    model.train_on_dataset(train_set, valid_set)
    logger.info("Saving model to %s" % out_file)
    model.save(out_file.as_posix())
    # Evaluate the model
    eval_result, plot_points = evaluate_model(model, valid_set)
    # Print evaluation results
    for key, val in eval_result.items():
        logger.info("  %s: %.4f", key, val)
    plot_and_save(plot_points, out_file.with_suffix(".png"))


if __name__ == "__main__":
    felix.init_logging("lightning_logs", True)
    args = parse_args()
    args.func(args)
