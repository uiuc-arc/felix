import logging
from pathlib import Path
from typing import Union

import pytorch_lightning as pl
import torch
from torch.utils import data
from tvm.auto_scheduler import cost_model as cm
from tvm.auto_scheduler.cost_model import MLPCostModel

from .utils import transpose2

__all__ = ["MLPCostModel", "MLPModelPLWrapper"]
logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class MLPModelPLWrapper(MLPCostModel, pl.LightningModule):
    def __init__(
        self,
        n_features: int = 154,
        loss_func: str = "log_mse",
        use_latency: bool = True,
        lr: float = 7e-4,
        wd: float = 1e-6,
        batch_size: int = 512,
    ) -> None:
        super().__init__(n_features, 256)
        self.loss_func_map = {
            "pair_cmp_acc": cm.metric_pairwise_cmp_acc,
            "peak_score@1": lambda x, y: cm.metric_peak_score(x, y, 1),
            "peak_score@5": lambda x, y: cm.metric_peak_score(x, y, 5),
        }
        if loss_func != "rank":
            self.loss_func_map["rank_loss"] = cm.MLPLossFunc("rank")
        self.main_loss, self.main_loss_f = loss_func, cm.MLPLossFunc(loss_func)
        self.lr = lr
        self.wd = wd
        self.batch_size = batch_size
        self.train_set = self.val_set = None
        self.use_latency = use_latency
        self.save_hyperparameters()

    @property
    def val_loss_name(self):
        return f"val/{self.main_loss}_loss"

    def set_dataset_(self, dataset: cm.SegmentDataset, split_ratio: float):
        dataset.use_latency = self.use_latency
        n_datum = len(dataset)
        n_train = int(n_datum * split_ratio)
        indices = torch.randperm(n_datum)
        self.train_set = data.Subset(dataset, indices[:n_train])  # type: ignore
        self.val_set = data.Subset(dataset, indices[n_train:])  # type: ignore
        logger.info(f"Loaded {(n_train, n_datum - n_train)} data points.")

    def training_step(self, batch, _):
        seg_sizes, features, labels, _ = batch
        output = self.forward_in_segments(seg_sizes, features)
        loss = self.main_loss_f(output, labels)
        self.log(f"train/{self.main_loss}_loss", loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, _):
        seg_sizes, features, labels, _ = batch
        output = self.forward_in_segments(seg_sizes, features)
        loss = self.main_loss_f(output, labels)
        self.log(self.val_loss_name, loss, batch_size=self.batch_size)
        for name, loss_f in self.loss_func_map.items():
            loss = loss_f(output, labels)
            self.log(f"val/{name}", loss, batch_size=self.batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": self.val_loss_name,
        }

    def train_dataloader(self) -> cm.BatchLoadingDataLoader:
        if self.train_set is None:
            raise ValueError("No training data")
        return cm.BatchLoadingDataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> cm.BatchLoadingDataLoader:
        if self.val_set is None:
            raise ValueError("No validation data")
        return cm.BatchLoadingDataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)

    def run_and_plot_validation(self, save_path: Path):
        from matplotlib import pyplot as plt

        data_points = []
        val_loader = self.val_dataloader()
        for seg_sizes, features, labels, _ in val_loader:
            pred_vals = self.forward_in_segments(seg_sizes, features)
            if self.main_loss == "log_mse":
                pred_vals = torch.exp(pred_vals)
            if self.use_latency:
                data_points.append((1 / pred_vals.detach(), 1 / labels))
            else:
                data_points.append((pred_vals.detach(), labels))
        x_log = self.main_loss in ("mse", "log_mse")
        xs, ys = transpose2(data_points)
        xs, ys = torch.cat(xs), torch.cat(ys)
        fig, ax = plt.subplots(1, 1)
        ax.scatter(xs, ys, s=1)
        label = "latency (sec)" if self.use_latency else "throughput (TFLOPs/sec)"
        if x_log:
            ax.set_xlabel(f"Predicted {label}")
            ax.set_xscale("log")
        else:
            ax.set_xlabel("Predicted throughput score")
        ax.set_ylabel(f"Measured {label}")
        ax.set_yscale("log")
        fig.savefig(save_path.as_posix(), dpi=300)

    def output_to_performance(self, flops, output):
        return super().output_to_performance(self.main_loss, self.use_latency, flops, output)
