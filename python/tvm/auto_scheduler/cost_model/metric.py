"""Evaluation metric for the cost models"""
from typing import Union

import numpy as np
from torch import Tensor
from tvm.autotvm.tuner.metric import max_curve

ArrayT = Union[np.ndarray, Tensor]
__all__ = [
    "metric_r_squared",
    "metric_rmse",
    "metric_pairwise_cmp_acc",
    "metric_top_k_recall",
    "metric_peak_score",
]


def _to_ndarray(x: ArrayT) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return x


def metric_r_squared(preds: ArrayT, labels: ArrayT):
    """Compute R^2 value"""
    s_tot = ((labels - labels.mean()) ** 2).sum()
    s_res = ((labels - preds) ** 2).sum()
    if s_tot < 1e-6:
        return 1
    return 1 - s_res / s_tot


def metric_rmse(preds: ArrayT, labels: ArrayT):
    """Compute RMSE (Rooted Mean Square Error)"""
    return np.sqrt(_to_ndarray(((preds - labels) ** 2).mean()))


def metric_pairwise_cmp_acc(preds: ArrayT, labels: ArrayT):
    """Compute the accuracy of pairwise comparision"""

    def vec_to_pair_com(vec: np.ndarray):
        return (vec.reshape((-1, 1)) - vec) > 0  # type: ignore

    n = len(preds)
    if n <= 1:
        return 0.5
    preds, labels = _to_ndarray(preds), _to_ndarray(labels)
    preds = vec_to_pair_com(preds)
    labels = vec_to_pair_com(labels)
    correct_ct = np.triu(np.logical_not(np.logical_xor(preds, labels)), k=1).sum()
    return correct_ct / (n * (n - 1) / 2)


def metric_top_k_recall(preds: ArrayT, labels: ArrayT, top_k: int):
    """Compute recall of top-k@k = |(top-k according to prediction) intersect (top-k according to ground truth)| / k."""
    preds, labels = _to_ndarray(preds), _to_ndarray(labels)
    real_top_k = set(np.argsort(-labels)[:top_k])
    predicted_top_k = set(np.argsort(-preds)[:top_k])
    recalled = real_top_k.intersection(predicted_top_k)
    return 1.0 * len(recalled) / top_k


def metric_peak_score(preds: ArrayT, labels: ArrayT, top_k: int):
    """Compute average peak score"""
    preds, labels = _to_ndarray(preds), _to_ndarray(labels)
    trials = np.argsort(preds)[::-1][:top_k]
    trial_scores = labels[trials]
    curve = max_curve(trial_scores) / np.max(labels)
    return np.mean(curve)
