import logging
from pathlib import Path
from typing import Dict, Union

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils import data
from tvm.auto_scheduler import cost_model as cm
from tvm.auto_scheduler.cost_model import MLPCostModel

__all__ = ["MLPCostModel", "MLPModelPLWrapper"]
logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class MLPModelPLWrapper(MLPCostModel, pl.LightningModule):
    def __init__(
        self,
        lr: float = 7e-4,
        wd: float = 1e-6,
        batch_size: int = 512,
        loss_func: str = "log_mse",
    ) -> None:
        other_metric_fs = {
            "pair_cmp_acc": cm.metric_pairwise_cmp_acc,
            "peak_score@1": lambda x, y: cm.metric_peak_score(x, y, 1),
            "peak_score@5": lambda x, y: cm.metric_peak_score(x, y, 5),
        }
        main_log_f = cm.MLPLossFunc(loss_func)
        if loss_func in ("mse", "log_mse"):
            other_metric_fs["rank_loss"] = cm.MLPLossFunc("rank")
        elif loss_func == "rank":
            other_metric_fs["MSE"] = cm.MLPLossFunc("mse")
            other_metric_fs["logMSE"] = cm.MLPLossFunc("log_mse")
        else:
            raise ValueError(f"Unknown loss function {loss_func}")
        super().__init__(154, 256, loss_func)
        self.loss_func_map: Dict[str, nn.Module] = other_metric_fs
        self.lr = lr
        self.wd = wd
        self.batch_size = batch_size
        self.loss_func = loss_func, main_log_f
        self.train_set = self.val_set = None
        self.save_hyperparameters()

    @property
    def val_loss_name(self):
        return f"val/{self.loss_func[0]}_loss"

    def add_dataset_(self, dataset: cm.SegmentDataset):
        n_datum = len(dataset)
        n_train = int(n_datum * 0.8)
        n_val = n_datum - n_train
        logger.info(f"Loaded {(n_train, n_val)} data points.")
        self.train_set, self.val_set = data.random_split(dataset, [n_train, n_val])

    def training_step(self, batch, _):
        seg_sizes, features, labels, _ = batch
        output = self.forward_in_segments(seg_sizes, features, inference=False)
        loss_f_name, loss_f = self.loss_func
        loss = loss_f(output, labels)
        self.log(f"train/{loss_f_name}", loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, _):
        seg_sizes, features, labels, _ = batch
        output = self.forward_in_segments(seg_sizes, features, inference=False)
        _, loss_f = self.loss_func
        loss = loss_f(output, labels)
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
