"""Anomaly-detection metrics: ROC AUC and detection-at-fixed-false-alarm.

Convention: label 1 = anomaly/fault, and a *higher* score means *more*
anomalous. For SignalMint the score is the model's negative log-likelihood --
faulty signals are improbable under a model trained only on healthy data.
"""

from __future__ import annotations

import numpy as np

__all__ = ["roc_auc", "detection_at_false_alarm"]


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (tie-aware)."""
    from scipy.stats import rankdata

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def detection_at_false_alarm(
    scores: np.ndarray, labels: np.ndarray, target_fa: float = 0.01
) -> tuple[float, float, float]:
    """Detection rate when the threshold is fixed to a target false-alarm rate.

    Args:
        scores: Per-item anomaly scores (higher = more anomalous).
        labels: 0 for normal, 1 for anomaly.
        target_fa: Desired fraction of normals flagged (false-alarm rate).

    Returns:
        ``(detection_rate, threshold, achieved_false_alarm)``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    neg = scores[labels == 0]
    pos = scores[labels == 1]
    if neg.size == 0 or pos.size == 0:
        return float("nan"), float("nan"), float("nan")
    # Threshold at the (1 - target_fa) quantile of normals; use strict exceedance
    # so that tied normal scores do not all count as false alarms.
    threshold = float(np.quantile(neg, 1.0 - target_fa))
    detection = float(np.mean(pos > threshold))
    achieved_fa = float(np.mean(neg > threshold))
    return detection, threshold, achieved_fa
