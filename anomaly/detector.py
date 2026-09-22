"""Orchestrate anomaly evaluation: NLL model vs autoencoder baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.dataset import FrameSet
from ..model.wavenet import WaveNetLite
from .metrics import detection_at_false_alarm, roc_auc
from .score import calibrate_threshold, frame_nll_scores

__all__ = ["AnomalyReport", "evaluate_anomaly"]


@dataclass
class AnomalyReport:
    """Anomaly-detection results for one scorer.

    Attributes:
        auc: ROC AUC over all frames.
        detection_rate: True-positive rate at the fixed false-alarm threshold.
        threshold: Score threshold used.
        achieved_fa: Realized false-alarm rate at that threshold.
        target_fa: Requested false-alarm rate.
        n_normal: Number of normal frames evaluated.
        n_fault: Number of fault frames evaluated.
    """

    auc: float
    detection_rate: float
    threshold: float
    achieved_fa: float
    target_fa: float
    n_normal: int
    n_fault: int


def _report(scores: np.ndarray, labels: np.ndarray, target_fa: float) -> AnomalyReport:
    auc = roc_auc(scores, labels)
    det, thr, fa = detection_at_false_alarm(scores, labels, target_fa)
    return AnomalyReport(
        auc=auc,
        detection_rate=det,
        threshold=thr,
        achieved_fa=fa,
        target_fa=target_fa,
        n_normal=int((labels == 0).sum()),
        n_fault=int((labels == 1).sum()),
    )


def evaluate_anomaly(
    model: WaveNetLite,
    eval_frames: FrameSet,
    target_fa: float = 0.01,
    batch_size: int = 64,
) -> tuple[AnomalyReport, np.ndarray]:
    """Score ``eval_frames`` with the model's NLL and compute anomaly metrics.

    Returns the report and the raw per-frame scores (for downstream comparison or
    threshold calibration).
    """
    scores = frame_nll_scores(model, eval_frames, batch_size=batch_size)
    report = _report(scores, eval_frames.labels, target_fa)
    return report, scores


def report_from_scores(
    scores: np.ndarray, labels: np.ndarray, target_fa: float = 0.01
) -> AnomalyReport:
    """Build an :class:`AnomalyReport` from precomputed scores (e.g. baseline)."""
    return _report(np.asarray(scores), np.asarray(labels), target_fa)
