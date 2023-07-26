import argparse
import logging
import random
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from tvm import felix
from tvm.auto_scheduler.cost_model import (
    DatasetBuilder,
    SegmentDataset,
    XGBModel,
    make_dataset_from_log_file,
)
from tvm.auto_scheduler.feature import (
    get_per_store_features_from_measure_pairs as get_ansor_features,
)
from tvm.felix import ffi, utils

WORK_DIR = Path(f"lightning_logs/{utils.TARGET.model}")
MRECORDS = WORK_DIR / "measure_records"
ALL_TASKS_PKL = Path("lightning_logs/tenset/network_info/all_tasks.pkl")
FELIX_TASKS_PKL_OUT = MRECORDS / "felix_tasks.pkl"
FELIX_DATASET_OUT = MRECORDS / "felix_dataset.pkl"
XGB_MODEL_OUT = WORK_DIR / "cost_models" / "xgb.pkl"
FelixTasks = List[Tuple[felix.SymTask, felix.TaskInstance]]

logger = logging.getLogger(__name__)


def parse_args():
    def varargs(args):
        d = vars(args)
        d.pop("func")
        d.pop("command")
        d.pop("seed")
        return d

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-t", "--tasks", type=Path, default=ALL_TASKS_PKL)

    commands = parser.add_subparsers(dest="command", required=True)
    fdataset = commands.add_parser("felix_dataset")
    fdataset.add_argument("-l", "--logs", required=True, nargs="+", type=str)
    fdataset.add_argument("-o", "--output", type=Path, default=FELIX_DATASET_OUT)
    fdataset.add_argument("--tasks-out", type=Path, default=FELIX_TASKS_PKL_OUT)
    fdataset.add_argument("--ansor-features", action="store_true")
    fdataset.set_defaults(func=lambda args: felix_dataset(**varargs(args)))

    tr_mlp = commands.add_parser("train_mlp")
    tr_mlp.add_argument("-d", "--dataset", required=True, type=Path)
    tr_mlp.add_argument("--train-ratio", type=float, default=0.9)
    tr_mlp.add_argument("--min-lat", type=float)
    tr_mlp.add_argument("--max-lat", type=float)
    tr_mlp.add_argument("--use-latency", action="store_true")
    tr_mlp.add_argument("--loss-f", type=str, choices=["mse", "log_mse", "rank"], default="log_mse")
    tr_mlp.set_defaults(func=lambda args: train_mlp(**varargs(args)))

    tr_xgb = commands.add_parser("train_xgb")
    tr_xgb.add_argument("-l", "--logs", required=True, nargs="+", type=str)
    tr_xgb.add_argument(
        "--split-scheme",
        type=str,
        choices=["by_task", "within_task", "by_target"],
        default="within_task",
    )
    tr_xgb.add_argument("--train-ratio", type=float, default=0.9)
    tr_xgb.add_argument("--sample-in-files", type=int)
    tr_xgb.add_argument("--min-sample-size", type=int, default=48)
    tr_xgb.add_argument("-o", "--output", type=Path, default=XGB_MODEL_OUT)
    tr_xgb.set_defaults(func=lambda args: train_xgb(**varargs(args)))

    return parser.parse_args()


def seed_all(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)


def load_ansor_configs_stream(tasks: FelixTasks, config_files: List[Path]):
    from tvm.auto_scheduler import RecordReader

    def config_stream(file) -> Iterator[utils.InpResPair]:
        for inp, res in RecordReader(str(file)):
            if res.error_no != 0:
                continue
            yield inp, res

    task_by_key = {inst.ansor_task.workload_key: idx for idx, (_, inst) in enumerate(tasks)}
    for file in config_files:
        # Have a little peek into the file to get the task info.
        # We're assuming that all configs in the file are for the same task.
        # This allows us to delay the actual loading of the configs until later
        # and significantly reduce the memory footprint.
        try:
            task = next(config_stream(file))[0].task
        except StopIteration:
            logger.warning(f"File {file} is empty. Skipping.")
            continue
        if (task_list_idx := task_by_key.get(task.workload_key)) is not None:
            sym_task, inst = tasks[task_list_idx]
            yield sym_task, inst, file, config_stream(file)


def _add_felix_configs(
    builder, sym_task: felix.SymTask, inst: felix.TaskInstance, inp_res_pairs, filename
):
    conf_counter = 0
    flops = sym_task.get_flops(inst.sizes)
    for backbone, configs_ in utils.group_configs_by_backbone(inp_res_pairs).items():
        sketch = sym_task.find_sketch(backbone)
        assert sketch is not None
        feature_f = sketch.fetch_features(inst.sizes, prime_factorize=False)
        assert feature_f is not None
        var_values = [dict(ffi.extract_config_dict(inp.state)) for inp, _ in configs_]
        our_feats, _ = feature_f.run_on_initial_configs(var_values)
        for (_, res), feats in zip(configs_, our_feats):
            nan_or_inf = torch.isnan(feats).any() or torch.isinf(feats).any()
            # Anything more than 100 (heuristically chosen) is probably a bug
            # because the features are already log-scaled.
            large_magnitude = torch.abs(feats).max() > 100
            if nan_or_inf or large_magnitude:
                continue
            cost = np.array([float(x) for x in res.costs]).mean()
            metadata = {
                "task_idx": inst.idx,
                "config_file": filename,
                "n_steps": len(backbone),
            }
            builder.add_config_(feats, flops, cost, metadata)
            conf_counter += 1
    return conf_counter


def felix_dataset(
    logs: List[Path], tasks: Path, output: Path, tasks_out: Path, ansor_features: bool
):
    if output.is_file():
        logger.warning(f"Found existing dataset at {output}. Skipping.")
        return
    tasks_ = felix.extract_tenset_pickle_tasks(tasks, tasks_out)
    logger.info(f"Found {len(logs)} json files.")
    builder = DatasetBuilder()
    for sym_task, inst, filename, configs in load_ansor_configs_stream(tasks_, logs):
        configs = list(configs)  # Only actually read in configs now.
        costs = [np.array([float(c) for c in res.costs]).mean() for _, res in configs]
        least5_lat = np.array(sorted(costs)[:5]) * 1e6
        logger.info(
            f"""Loaded task {inst.idx} ({sym_task})
  with sizes {inst.sizes}
  Top 5 configs (latency in us): {least5_lat}"""
        )
        if ansor_features:
            inputs, results = utils.transpose2(configs)
            flops = sym_task.get_flops(inst.sizes)
            features, _, _ = get_ansor_features(inputs, results)
            features = [torch.tensor(f.astype(float)).float() for f in features]
            builder.add_configs_(features, flops, torch.tensor(costs), {"config_file": filename})
            logger.info("Added %d configs for task", len(configs))
        else:
            conf_counter = _add_felix_configs(builder, sym_task, inst, configs, filename)
            logger.info("Added %d / %d configs for task", conf_counter, len(configs))
    torch.save(builder.to_dataset(), output)


def ansor_dataset(
    logs: List[Path],
    output: Path,
    sample_in_files: int,
    min_sample_size: int,
    tasks: Path,
):
    logger.info("Load tasks...")
    utils.load_and_register_ansor_tasks(tasks, True)
    if sample_in_files:
        logs = random.sample(logs, sample_in_files)
    logger.info("Featurize measurement records...")
    output.parent.mkdir(parents=True, exist_ok=True)
    make_dataset_from_log_file(logs, output, min_sample_size)


def train_xgb(
    logs: List[Path],
    split_scheme: str,
    train_ratio: float,
    sample_in_files: int,
    min_sample_size: int,
    tasks: Path,
    output: Path,
):
    logger.info("Load tasks...")
    utils.load_and_register_ansor_tasks(tasks, True)
    if sample_in_files:
        logs = random.sample(logs, sample_in_files)
    logger.info("Featurize measurement records...")
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = make_dataset_from_log_file(logs, min_sample_size)
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
    logger.info("Saving model to %s" % output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output.as_posix())
    # Evaluate the model
    eval_result, plot_points = evaluate_xgb(model, valid_set)
    # Print evaluation results
    for key, val in eval_result.items():
        logger.info("  %s: %.4f", key, val)
    plot_xgb_perf(plot_points, output.with_suffix(".png"))


def evaluate_xgb(model: XGBModel, valid_set):
    from tvm.auto_scheduler.cost_model import (
        metric_pairwise_cmp_acc,
        metric_peak_score,
        metric_r_squared,
        metric_rmse,
    )

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


def plot_xgb_perf(data_points: list, out_file: Path):
    fig, ax = plt.subplots(1, 1)
    xs, ys = zip(*data_points)
    ax.scatter(xs, ys, s=1)
    ax.set_xlabel("Predicted throughput score")
    ax.set_ylabel("Measured throughput score")
    fig.savefig(out_file.as_posix(), dpi=300)
    plt.close(fig)


def plot_configs_perf_dist(dataset: SegmentDataset, save_to: Path):
    fig, ax = plt.subplots(dpi=300)
    xs = torch.arange(len(dataset))
    ax.scatter(xs.numpy(), dataset.flops_lats[:, 1].numpy(), s=1)
    ax.set_ylabel("Measured latency (sec)")
    ax.set_yscale("log")
    fig.savefig(save_to.as_posix())


def select_dataset_by_lat(
    dataset: SegmentDataset, min_: Optional[float] = None, max_: Optional[float] = None
):
    latencies = dataset.flops_lats[:, 1]
    all_true = torch.ones_like(latencies, dtype=torch.bool)
    lb_mask = all_true if min_ is None else latencies >= min_
    ub_mask = all_true if max_ is None else latencies <= max_
    taken_indices = torch.nonzero(lb_mask & ub_mask)[:, 0]
    logger.info(f"Selected {len(taken_indices)} configs out of {len(dataset)}")
    return dataset.slice(taken_indices)


def train_mlp(
    dataset: Path,
    loss_f: str,
    train_ratio: float,
    min_lat: Optional[float],
    max_lat: Optional[float],
    use_latency: bool,
    tasks,
):
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint

    dataset_ = torch.load(dataset)
    assert isinstance(dataset_, SegmentDataset)
    # We don't need this and manipulating it is slow; setting it to None will speed up the training.
    dataset_.conf_meta = None
    pred_model = felix.MLPModelPLWrapper(dataset_.features.shape[1], loss_f, use_latency)
    pred_model.set_dataset_(select_dataset_by_lat(dataset_, min_lat, max_lat), train_ratio)
    val_loss = pred_model.val_loss_name
    filename = "epoch={epoch:02d}-loss={%s:.4f}" % val_loss
    ckpt = ModelCheckpoint(
        monitor=val_loss, save_top_k=1, filename=filename, auto_insert_metric_name=False
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=200,
        gradient_clip_val=0.5,
        callbacks=[ckpt],
    )
    trainer.fit(pred_model)
    trainer.validate(pred_model)
    log_dir = trainer.log_dir
    assert log_dir is not None
    plot_configs_perf_dist(dataset_, Path(log_dir) / "configs_perf_dist.png")
    pred_model.load_from_checkpoint(ckpt.best_model_path)
    pred_model.run_and_plot_validation(Path(log_dir) / "val_set_pred.png")


if __name__ == "__main__":
    args = parse_args()
    seed_all(args.seed)
    felix.init_logging("lightning_logs", True)
    args.func(args)
