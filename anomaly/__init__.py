"""Anomaly-detection head.

Phase 2: negative-log-likelihood scoring, threshold calibration on normal data,
and detection@fixed-false-alarm reporting against an autoencoder baseline.
"""

from __future__ import annotations

from .baseline import autoencoder_scores, train_autoencoder
from .detector import AnomalyReport, evaluate_anomaly, report_from_scores
from .metrics import detection_at_false_alarm, roc_auc
from .score import calibrate_threshold, frame_nll_scores

__all__ = [
    "roc_auc",
    "detection_at_false_alarm",
    "frame_nll_scores",
    "calibrate_threshold",
    "evaluate_anomaly",
    "report_from_scores",
    "AnomalyReport",
    "train_autoencoder",
    "autoencoder_scores",
]
