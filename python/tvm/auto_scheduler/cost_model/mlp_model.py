import logging
from typing import Dict, List, Union

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
        lf = checkpoint["hyper_parameters"]["loss_func"]
        n_feats = checkpoint["hyper_parameters"]["n_features"]
        self.model = MLPCostModel(n_feats, 256, lf == "log_mse", lf != "rank")
        self.loss_f = MLPLossFunc(lf)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=7e-4, weight_decay=1e-6)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model = self.model.eval().cuda()
        self.dataset = DatasetBuilder()

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
        self.dataset.add_configs_(features, flops, lats)

        # pred = self.model.forward_on_batch(features, False)
        # loss = self.loss_f(pred, thruputs)
        # self.optim.zero_grad()
        # loss.backward()
        # self.optim.step()

        print_per_epoches = 5
        n_epoch = 30
        early_stop = 5
        lr = 7e-4
        wd = 1e-6
        grad_clip = 0.5
        batch_size = 512

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        best_train_loss, best_epoch = 1e10, 0
        dataset = self.dataset.to_dataset()
        dataloader = BatchLoadingDataLoader(dataset, batch_size, shuffle=True)
        logger.info("Dataset size: %d (%d batches)", len(dataset), len(dataloader))
        for epoch in range(n_epoch):
            epoch_loss = 0.0
            for segment_sizes, features, labels, _ in dataloader:
                segment_sizes = segment_sizes.cuda()
                features = features.cuda()
                labels = labels.cuda()
                optimizer.zero_grad()
                loss = self.loss_f(
                    self.model.forward_in_segments(segment_sizes, features, False), labels
                )
                loss.backward()
                optimizer.step()
                epoch_loss = moving_average(epoch_loss, loss.item())
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)  # type: ignore
            if epoch % print_per_epoches == 0 or epoch == n_epoch - 1:
                logger.info("Epoch: %d\tEpoch train Loss: %.4f", epoch, epoch_loss)
            # Early stop
            if epoch_loss < best_train_loss:
                best_train_loss = epoch_loss
                best_epoch = epoch
            elif epoch - best_epoch >= early_stop:
                logger.info("Early stop. Current epoch: %d; best epoch: %d", epoch, best_epoch)
                break

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


class MLPCostModel(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        output_log: bool,
        is_throughput: bool,
        out_dim: int = 1,
        use_norm: bool = False,
    ):
        super().__init__()
        self.output_log = output_log
        self.is_throughput = is_throughput
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
        return self.forward_in_segments(segment_sizes, features.flatten(0, 1), True)

    def forward_on_batch(self, features: List[Tensor], inference: bool = True):
        segment_sizes = [len(f) for f in features]
        segment_sizes = torch.tensor(segment_sizes, dtype=torch.long, device=self.device)
        features_ = torch.cat(features, dim=0)
        return self.forward_in_segments(segment_sizes, features_, inference)

    def forward_in_segments(self, segment_sizes, features, inference: bool):
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
        output = self.decoder(output).squeeze(-1)
        if self.output_log and inference:
            output = torch.exp(output)
        return output


class DatasetBuilder:
    def __init__(self) -> None:
        self.features: List[Tensor] = []
        self.labels: List[float] = []
        self.conf_meta: List[dict] = []

    def add_config_(self, feature: Tensor, flops: Num, latency: Num, conf_meta: dict):
        if (label := self.make_label(flops, latency)) <= 0:
            return
        self.features.append(feature)
        self.labels.append(label)
        self.conf_meta.append(conf_meta)

    def add_configs_(self, features: list, flops: float, latencies: Tensor):
        labels = self.make_label(flops, latencies)
        for feat_, label_ in zip(features, labels):
            if label_ > 0:
                self.features.append(feat_)
                self.labels.append(label_.item())

    def to_dataset(self):
        seg_size = torch.tensor([f.shape[0] for f in self.features])
        conf_meta = self.conf_meta if self.conf_meta else [{} for _ in self.features]
        return SegmentDataset(
            seg_size, torch.cat(self.features, dim=0), torch.tensor(self.labels), conf_meta
        )

    def make_label(self, flops, lat_sec):
        # output from model is trained to be throughput (TFlops/s)
        return flops / lat_sec / 1e12


class SegmentDataset(data.Dataset):
    def __init__(
        self,
        segment_sizes: Tensor,
        features: Tensor,
        labels: Tensor,
        conf_meta: List[Dict],
    ):
        self.features = features
        seg_sum = torch.cumsum(segment_sizes, 0, dtype=torch.int32)
        begins, ends = seg_sum - segment_sizes, seg_sum
        self.segment_ranges = torch.stack([begins, ends], dim=1)
        self.labels = labels
        self.conf_meta = conf_meta

    def __getitem__(self, index_: Union[int, List[int]]):
        if isinstance(index_, Tensor):
            index = index_
        else:
            index = torch.tensor(index_)
        ranges = self.segment_ranges[index]  # 0d or 1d tensor
        labels = self.labels[index]
        conf: Union[Dict, List[Dict]]
        if index.ndim == 0:
            begin, end = ranges
            seg_size = end - begin
            features = self.features[begin:end]
            conf = self.conf_meta[int(index.item())]
        else:
            begin, end = ranges.T
            seg_size = end - begin
            features = torch.cat([self.features[b:e] for b, e in ranges], dim=0)
            conf = [self.conf_meta[n] for n in index.tolist()]
        return seg_size, features, labels, conf

    def __len__(self):
        return len(self.labels)


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
            yield self.dataset[batch_indices]

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
