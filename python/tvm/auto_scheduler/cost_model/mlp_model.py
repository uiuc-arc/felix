import logging
from dataclasses import dataclass
from functools import total_ordering
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.utils import data

from .. import feature as ft
from ..loop_state import State
from .cost_model import PythonBasedModel

logger = logging.getLogger(__name__)
__all__ = [
    "AnsorMLPModel",
    "MLPCostModel",
    "MLPLossFunc",
    "DatasetBuilder",
    "SegmentDataset",
    "BatchLoadingDataLoader",
]
Num = Union[int, float]


def moving_average(average, update):
    if average is None:
        return update
    else:
        return average * 0.95 + update * 0.05


class AnsorMLPModel(PythonBasedModel):
    def __init__(self, model_path: str) -> None:
        super().__init__()
        checkpoint = torch.load(model_path, map_location="cpu")
        self.loss_func = checkpoint["hyper_parameters"]["loss_func"]
        self.use_latency = checkpoint["hyper_parameters"]["use_latency"]
        n_feats = checkpoint["hyper_parameters"]["n_features"]
        self.model = MLPCostModel(n_feats, 256)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model = self.model.eval().cuda()
        self.data_builder = DatasetBuilder()

    def update(self, inputs, results):
        keys = set(input_.task.workload_key for input_ in inputs)
        if len(keys) != 1:
            logger.warning(f"Update with multiple tasks (workload keys = {keys}) is unsupported")
            return
        task = inputs[0].task
        flops = float(task.compute_dag.flop_ct)
        states = [input_.state for input_ in inputs]
        features = ft.get_per_store_features_from_states(states, task)
        features = [torch.from_numpy(st_feats.astype(float)).float() for st_feats in features]
        lats = torch.tensor([[float(x) for x in res.costs] for res in results]).mean(dim=1)
        self.data_builder.add_configs_(features, flops, lats, {})
        self.model.train_self(
            self.data_builder.to_dataset(self.use_latency),
            self.loss_func,
            lr=7e-4,
            weight_decay=1e-6,
            batch_size=512,
            n_epoch=30,
            early_stop=5,
        )

    def predict(self, task, states: List[State]) -> List[float]:
        # `features`` is a sequence of np.ndarray each with [n_buf, n_feature(=154)]
        # (It's a object-typed numpy array of numpy arrays)
        features = ft.get_per_store_features_from_states(states, task)
        features = [
            torch.from_numpy(st_feats.astype(float)).float().cuda() for st_feats in features
        ]
        return self.model.forward_on_batch(features).detach().tolist()

    def predict_stages(self, task, states: List[State]):
        raise RuntimeError("MLPModel does not support predict_stages")


@dataclass
@total_ordering
class LatAndThruput:
    latency_us: float
    thruput_tflops: float

    @classmethod
    def from_thruput(cls, flops: float, thruput_tflops: float):
        thruput_flops = thruput_tflops * 1e12
        latency_us = flops / thruput_flops * 1e6
        return cls(latency_us, thruput_tflops)

    @classmethod
    def from_latency_s(cls, flops: float, latency_s: float):
        latency_us = latency_s * 1e6
        thruput_tflops = flops / latency_s / 1e12
        return cls(latency_us, thruput_tflops)

    def __lt__(self, other) -> bool:
        if not isinstance(other, LatAndThruput):
            return NotImplemented
        return self.thruput_tflops < other.thruput_tflops

    def __repr__(self) -> str:
        return f"({self.latency_us:.2f} us, {self.thruput_tflops:.4f} TFLOPs)"


@dataclass
@total_ordering
class PerfScore:
    score: float

    def __lt__(self, other) -> bool:
        if not isinstance(other, PerfScore):
            return NotImplemented
        return self.score < other.score

    def __repr__(self) -> str:
        return f"PerfScore({self.score:.3f})"


Performance = Union[LatAndThruput, PerfScore]


class MLPCostModel(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim: int = 1,
        use_norm: bool = False,
    ):
        super().__init__()
        self.segment_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.norm = nn.BatchNorm1d(hidden_dim) if use_norm else nn.Identity()
        self.l0 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.l1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Linear(hidden_dim, out_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, features: Tensor):
        segment_sizes = torch.full(
            (features.shape[0],), features.shape[1], dtype=torch.long, device=self.device
        )
        return self.forward_in_segments(segment_sizes, features.flatten(0, 1))

    def forward_on_batch(self, features: List[Tensor]):
        segment_sizes = [len(f) for f in features]
        segment_sizes = torch.tensor(segment_sizes, dtype=torch.long, device=self.device)
        features_ = torch.cat(features, dim=0)
        return self.forward_in_segments(segment_sizes, features_)

    def forward_in_segments(self, segment_sizes, features):
        n_seg = segment_sizes.shape[0]
        device = self.device
        segment_sizes = segment_sizes.long()
        features = self.segment_encoder(features)
        segment_indices = torch.repeat_interleave(torch.arange(n_seg, device=device), segment_sizes)
        n_dim = features.shape[1]
        segment_sum = torch.scatter_add(
            torch.zeros((n_seg, n_dim), dtype=features.dtype, device=device),
            0,
            segment_indices.view(-1, 1).expand(-1, n_dim),
            features,
        )
        output = self.norm(segment_sum)
        output = self.l0(output) + output
        output = self.l1(output) + output
        return self.decoder(output).squeeze(-1)

    def output_to_performance(self, loss_func: str, use_latency: bool, flops: float, output: float):
        import numpy as np

        if loss_func == "rank":
            return PerfScore(output)
        elif loss_func == "log_mse":
            output = np.exp(output)
        if use_latency:
            return LatAndThruput.from_latency_s(flops, 1 / output)
        else:
            return LatAndThruput.from_thruput(flops, output)

    def train_self(
        self,
        dataset: "SegmentDataset",
        loss_func: str,
        lr: float,
        weight_decay: float,
        batch_size: int,
        n_epoch: int,
        early_stop: int,
        grad_clip: float = 0.5,
        print_per_epoches: int = 5,
    ):
        best_train_loss, best_epoch = 1e10, 0
        original_device = self.device
        self.train().cuda()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        dataloader = BatchLoadingDataLoader(dataset, batch_size, shuffle=True)
        logger.info("Dataset size: %d (%d batches)", len(dataset), len(dataloader))
        loss_f = MLPLossFunc(loss_func)
        for epoch in range(n_epoch):
            epoch_loss = 0.0
            for segment_sizes, features, labels, _ in dataloader:
                segment_sizes = segment_sizes.cuda()
                features = features.cuda()
                labels = labels.cuda()
                optimizer.zero_grad()
                loss = loss_f(self.forward_in_segments(segment_sizes, features), labels)
                loss.backward()
                optimizer.step()
                epoch_loss = moving_average(epoch_loss, loss.item())
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)  # type: ignore
            if epoch % print_per_epoches == 0 or epoch == n_epoch - 1:
                logger.info("Epoch: %d\tEpoch train Loss: %.4f", epoch, epoch_loss)
            # Early stop
            if epoch_loss < best_train_loss:
                best_train_loss = epoch_loss
                best_epoch = epoch
            elif epoch - best_epoch >= early_stop:
                logger.info("Early stop. Current epoch: %d; best epoch: %d", epoch, best_epoch)
                break
        self.eval().to(original_device)


class DatasetBuilder:
    def __init__(self) -> None:
        self.features: List[Tensor] = []
        self.labels: List[Tuple[float, float]] = []
        self.conf_meta: List[dict] = []

    def add_config_(self, feature: Tensor, flops: Num, latency: Num, conf_meta: dict):
        self.features.append(feature)
        self.labels.append((flops, latency))
        self.conf_meta.append(conf_meta)

    def add_configs_(self, features: list, flops: Num, latencies: Tensor, conf_meta: dict):
        for feats, latency in zip(features, latencies):
            if not latency > 0:
                continue
            self.features.append(feats)
            self.labels.append((flops, latency.item()))
            self.conf_meta.append(conf_meta)

    def to_dataset(self, use_latency: bool = True):
        seg_size = torch.tensor([f.shape[0] for f in self.features])
        conf_meta = self.conf_meta if self.conf_meta else [{} for _ in self.features]
        features = torch.cat(self.features, dim=0)
        return SegmentDataset(seg_size, features, torch.tensor(self.labels), conf_meta, use_latency)


class SegmentDataset(data.Dataset):
    def __init__(
        self,
        segment_sizes: Tensor,
        features: Tensor,
        flops_lats: Tensor,
        conf_meta: Optional[List[Dict]],
        use_latency: bool = True,
    ):
        super().__init__()
        self.features = features
        seg_sum = torch.cumsum(segment_sizes, 0, dtype=torch.int32)
        begins, ends = seg_sum - segment_sizes, seg_sum
        self.segment_ranges = torch.stack([begins, ends], dim=1)
        self.flops_lats = flops_lats
        self.conf_meta = conf_meta
        self.use_latency = use_latency

    def _slice(self, indices: Tensor):
        ranges = self.segment_ranges[indices]
        begin, end = ranges.T
        segment_sizes = end - begin
        features = torch.cat([self.features[b:e] for b, e in ranges], dim=0)
        flops_lats = self.flops_lats[indices]
        conf_meta = [self.conf_meta[n] for n in indices.tolist()] if self.conf_meta else None
        return segment_sizes, features, flops_lats, conf_meta

    def slice(self, indices: Tensor):
        return SegmentDataset(*self._slice(indices))

    def __getitem__(self, index_: Union[int, Tensor]):
        squeeze = isinstance(index_, int)
        indices = torch.tensor([index_]) if squeeze else index_
        seg_size, features, fls, conf_meta = self._slice(indices)
        if self.use_latency:
            labels = 1 / fls[:, 1]  # Inverse of latency (sec) to maintain higher-better
        else:
            labels = fls[:, 0] / fls[:, 1] / 1e12  # In TFLOPs/sec
        if squeeze:
            seg_size, fls, features = seg_size.item(), fls.squeeze(0), features.squeeze(0)
            conf_meta = conf_meta[0] if conf_meta else None
        return seg_size, features, labels, conf_meta

    def __len__(self):
        return len(self.flops_lats)


class BatchLoadingDataLoader:
    def __init__(self, dataset, batch_size: int, shuffle: bool):
        self.dataset = dataset
        self.batch_size = batch_size
        if shuffle:
            self.sampler = data.RandomSampler(dataset)
        else:
            self.sampler = data.SequentialSampler(dataset)
        self.batch_sampler = data.BatchSampler(self.sampler, batch_size, False)

    def __iter__(self):
        for batch_indices in iter(self.batch_sampler):
            yield self.dataset[torch.tensor(batch_indices)]

    def sample_batch(self, batch_size):
        raise NotImplementedError()

    def __len__(self):
        from math import ceil

        return ceil(len(self.dataset) / self.batch_size)


class MLPLossFunc(nn.Module):
    def __init__(self, loss_func: str) -> None:
        super().__init__()
        if loss_func == "log_mse":
            self.forward = self.log_mse_loss
        elif loss_func == "mse":
            self.forward = self.mse_loss
        elif loss_func == "rank":
            self.forward = self.rank_net_loss
        else:
            raise ValueError("Invalid loss function: " + loss_func)

    def mse_loss(self, preds, labels):
        return nn.functional.mse_loss(preds, labels)

    def log_mse_loss(self, preds, labels):
        assert torch.all(labels > 0)
        labels = torch.log(labels)
        return nn.functional.mse_loss(preds, labels)

    def rank_net_loss(self, preds: Tensor, labels: Tensor):
        preds = torch.clamp(preds, -10, 10)
        s_ij = preds - preds.unsqueeze(1)
        p_ij = 1 / (torch.exp(s_ij) + 1)
        preds_prob = torch.triu(p_ij, diagonal=1)
        labels_prob = torch.triu((labels.unsqueeze(1) > labels).float(), diagonal=1)
        return nn.functional.binary_cross_entropy(preds_prob, labels_prob)
