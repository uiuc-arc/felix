import logging
import pickle as pkl
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import tvm
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
from tvm import auto_scheduler as ansor
from tvm import felix
from tvm.auto_scheduler.cost_model import DatasetBuilder as AnsorDatasetBuilder
from tvm.auto_scheduler.cost_model import SegmentDataset
from tvm.felix import utils
from tvm.felix.utils import AnsorTaskWeight, InpResPair

_logger = logging.getLogger(__file__)
PathLike = Union[str, Path]
FelixTasks = List[Tuple[felix.SymTask, felix.TaskInstance]]


def extract_tenset_pickle_tasks(task_pkl: PathLike):
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
        return felix.TaskInstance(inst.idx, inst.weight, inst.sizes, ansor_task)

    def patch_instances(sym_task, instances):
        return [
            inst_ for inst in instances if (inst_ := patch_one_instance(sym_task, inst)) is not None
        ]

    with open(task_pkl, "rb") as f:
        tasks = pkl.load(f)
    assert isinstance(tasks, list) and len(tasks) > 0
    tasks = [check_pair(task) for task in tasks]
    # Register workloads & override hardware params
    for i in range(len(tasks)):
        task, weight = tasks[i]
        ansor.workload_registry.register_workload_tensors(
            task.workload_key, task.compute_dag.tensors
        )
        task = ansor.SearchTask(
            workload_key=task.workload_key, target=felix.TARGET, hardware_params=felix.HW_PARAMS
        )
        tasks[i] = task, weight
    # Don't check the hash value of generated tasks against Tenset's.
    # This forfeits a key safety check but is necessary as Tenset's DAGs
    # are generated with TVM of an older version.
    return [
        felix.SymTaskAndInstances(sym_task, patch_instances(sym_task, instances))
        for sym_task, instances in felix.batch_create_tasks(
            tasks, hash_match=False, print_tasks=False
        )
    ]


def load_ansor_configs_stream(tasks: FelixTasks, config_files: List[Path]):
    from tvm.auto_scheduler import RecordReader

    def config_stream(file) -> Iterator[InpResPair]:
        for inp, res in RecordReader(str(file)):
            if res.error_no != 0:
                continue
            yield inp, res

    task_by_key = {inst.ansor_task.workload_key: idx for idx, (_, inst) in enumerate(tasks)}
    ret: List[Tuple[felix.SymTask, felix.TaskInstance, Iterator[InpResPair]]] = []
    for file in config_files:
        # Have a little peek into the file to get the task info.
        # We're assuming that all configs in the file are for the same task.
        # This allows us to delay the actual loading of the configs until later
        # and significantly reduce the memory footprint.
        try:
            task = next(config_stream(file))[0].task
        except StopIteration:
            _logger.warning(f"File {file} is empty. Skipping.")
            continue
        if (task_list_idx := task_by_key.get(task.workload_key)) is not None:
            sym_task, inst = tasks[task_list_idx]
            ret.append((sym_task, inst, config_stream(file)))
    return ret


class DatasetBuilder(AnsorDatasetBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.task_idx = self.sym_task = self.sizes = self.flops = None
        self.conf_counter = 0

    def set_task_(self, sym_task: felix.SymTask, inst: felix.TaskInstance):
        self.sym_task = sym_task
        self.task_idx = inst.idx
        self.sizes = inst.sizes
        assert self.sym_task is not None
        self.flops = self.sym_task.get_flops(self.sizes)
        self.conf_counter = 0

    def add_config_(self, config: dict, features: torch.Tensor):
        assert self.sym_task is not None
        assert self.sizes is not None
        assert self.task_idx is not None
        assert self.flops is not None
        metadata = {
            "task_name": str(self.sym_task),
            "task_idx": self.task_idx,
            "n_steps": len(config["backbone"]),
        }
        super().add_config_(features, self.flops, config["time"], metadata)
        self.conf_counter += 1


def configs_with_our_feats_(builder: DatasetBuilder, sym_task: felix.SymTask, sizes, configs):
    for backbone, backbone_confs in utils.group_configs_by_backbone(configs).items():
        sketch = sym_task.find_sketch(backbone)
        if sketch is None:
            _logger.warning(
                "Backbone %s of concrete configs is not found among symbolic ones.",
                backbone,
            )
            continue
        var_values = [c["var_values"] | sizes for c in backbone_confs]
        feature_f = sketch.fetch_features(sizes, prime_factorize=False)
        if feature_f is None:
            _logger.warning(
                "Feature extraction failed for task %s and backbone %s",
                str(sym_task),
                backbone,
            )
            continue
        our_feats, _ = feature_f.run_on_initial_configs(var_values)
        for conf, feats in zip(backbone_confs, our_feats):
            nan_or_inf = torch.isnan(feats).any() or torch.isinf(feats).any()
            # Anything more than 100 (heuristically chosen) is probably a bug
            # because the features are already log-scaled.
            large_magnitude = torch.abs(feats).max() > 100
            if nan_or_inf or large_magnitude:
                continue
            builder.add_config_(conf, feats)


def load_configs(tasks: FelixTasks, config_jsons: List[Path]):
    builder = DatasetBuilder()
    for sym_task, inst, configs in load_ansor_configs_stream(tasks, config_jsons):
        _logger.info(f"Loading task {inst.idx} ({sym_task})")
        try:
            # Only actually read in the configs now.
            loaded_confs = utils.process_inp_res_pairs(list(configs), False)
        except tvm.TVMError as e:
            _logger.warning(f"Failed to load configs for task; skipping. Error: {e}")
            continue
        builder.set_task_(sym_task, inst)
        configs_with_our_feats_(builder, sym_task, inst.sizes, loaded_confs)
        _logger.info("Added %d / %d configs for task", builder.conf_counter, len(loaded_confs))
    return builder.to_dataset()


def make_dataset(tasks_pkl: Path):
    work_dir = Path("lightning_logs") / felix.TARGET.model / "measure_records"
    feat_pkl = work_dir / f"{tasks_pkl.stem}_features.pkl"
    felix_tasks_pkl = work_dir / tasks_pkl.name
    if feat_pkl.exists():
        dataset = torch.load(feat_pkl)
        assert isinstance(dataset, SegmentDataset)
        return dataset
    if felix_tasks_pkl.exists():
        tasks = felix.load_and_register_tasks_(felix_tasks_pkl)
    else:
        tasks_ = extract_tenset_pickle_tasks(tasks_pkl)
        with open(felix_tasks_pkl, "wb") as f:
            pkl.dump(tasks_, f)
        tasks = [(sym_task, inst) for sym_task, instances in tasks_ for inst in instances]
    conf_jsons = list(work_dir.rglob("**/*.json"))
    _logger.info(f"Found {len(conf_jsons)} json files.")
    dataset = load_configs(tasks, conf_jsons)
    torch.save(dataset, feat_pkl)
    return dataset


def run_mlp_and_plot(pred_model: felix.MLPModelPLWrapper, save_dir: Path):
    data_points = []
    val_loader = pred_model.val_dataloader()
    for seg_sizes, features, labels, _ in val_loader:
        pred_vals = pred_model.forward_in_segments(seg_sizes, features, True)
        pred_vals = pred_vals.detach().numpy()
        for pred_val, groundtruth in zip(pred_vals, labels):
            groundtruth = groundtruth.item()
            data_points.append((pred_val, groundtruth))
    plot_and_save(data_points, save_dir, pred_model.loss_func[0])
    return data_points


def plot_and_save(data_points: list, out_file: Path, loss_kind: str):
    x_log = loss_kind in ("mse", "log_mse")
    fig, ax = plt.subplots(1, 1)
    xs, ys = utils.transpose2(data_points)
    ax.scatter(xs, ys, s=1)
    if x_log:
        ax.set_xlabel("Predicted throughput (TFLOPs/sec)")
        ax.set_xscale("log")
    else:
        ax.set_xlabel("Predicted throughput score")
    ax.set_ylabel("Measured throughput (TFLOPs/sec)")
    ax.set_yscale("log")
    fig.savefig(out_file.as_posix(), dpi=300)


def plot_configs_perf_dist(dataset: SegmentDataset, save_to: Path):
    fig, ax = plt.subplots(dpi=300)
    xs = torch.arange(len(dataset))
    ax.scatter(xs.numpy(), dataset.labels.numpy(), s=1)
    ax.set_ylabel("Measured throughput (TFLOPs/sec)")
    ax.set_yscale("log")
    fig.savefig(save_to.as_posix())


def select_dataset_by_perf(
    dataset: SegmentDataset, min_: Optional[float] = None, max_: Optional[float] = None
):
    all_true = torch.ones_like(dataset.labels, dtype=torch.bool)
    lb_mask = all_true if min_ is None else dataset.labels >= min_
    ub_mask = all_true if max_ is None else dataset.labels <= max_
    taken_indices = torch.nonzero(lb_mask & ub_mask)[:, 0]
    _logger.info(f"Selected {len(taken_indices)} configs out of {len(dataset)}")
    seg_size, features, labels, conf = dataset[taken_indices]
    return SegmentDataset(seg_size, features, labels, conf)  # type: ignore


def select_per_task_top_ratio(dataset: SegmentDataset, ratio: float):
    from collections import defaultdict

    assert 0 < ratio <= 1
    task_indices = defaultdict(list)
    for idx in range(len(dataset.conf_meta)):
        task_indices[dataset.conf_meta[idx]["task_idx"]].append(idx)
    taken_indices = []
    for indices in task_indices.values():
        indices = torch.tensor(indices)
        sorted_indices = indices[dataset.labels[indices].argsort(descending=True)]
        top_k = int(len(indices) * ratio)
        taken_indices.append(sorted_indices[:top_k])
    taken_indices = torch.cat(taken_indices)
    _logger.info(f"Selected {len(taken_indices)} configs out of {len(dataset)}")
    seg_size, features, labels, conf = dataset[taken_indices]
    return SegmentDataset(seg_size, features, labels, conf)  # type: ignore


def main():
    felix.init_logging("./lightning_logs", verbose_logging=True)
    task_pkl = Path("lightning_logs/tenset/network_info/all_tasks.pkl")

    pred_model = felix.MLPModelPLWrapper(loss_func="rank")
    dataset = select_per_task_top_ratio(make_dataset(task_pkl), ratio=0.5)
    pred_model.add_dataset_(dataset)
    val_loss = pred_model.val_loss_name
    filename = "epoch={epoch:02d}-loss={%s:.4f}" % val_loss
    ckpt = ModelCheckpoint(
        monitor=val_loss, save_top_k=1, filename=filename, auto_insert_metric_name=False
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=100,
        gradient_clip_val=0.5,
        callbacks=[ckpt],
    )
    trainer.fit(pred_model)
    trainer.validate(pred_model)
    log_dir = trainer.log_dir
    assert log_dir is not None
    plot_configs_perf_dist(dataset, Path(log_dir) / "configs_perf_dist.png")
    pred_model.load_from_checkpoint(ckpt.best_model_path)
    run_mlp_and_plot(pred_model, Path(log_dir) / "val_set_pred.png")


if __name__ == "__main__":
    main()
