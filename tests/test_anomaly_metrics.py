"""Tests for anomaly metrics (no torch needed)."""

from __future__ import annotations

import numpy as np

from signalmint.anomaly.metrics import detection_at_false_alarm, roc_auc


def test_roc_auc_perfect_separation() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.9, 1.0, 1.1])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert roc_auc(scores, labels) == 1.0


def test_roc_auc_random_is_half() -> None:
    scores = np.array([0.0, 1.0, 0.0, 1.0])
    labels = np.array([0, 1, 1, 0])
    assert roc_auc(scores, labels) == 0.5


def test_roc_auc_degenerate_single_class() -> None:
    assert np.isnan(roc_auc(np.array([1.0, 2.0]), np.array([0, 0])))


def test_detection_at_false_alarm_perfect() -> None:
    scores = np.concatenate([np.zeros(100), np.ones(50)])
    labels = np.concatenate([np.zeros(100), np.ones(50)]).astype(int)
    det, thr, fa = detection_at_false_alarm(scores, labels, target_fa=0.01)
    assert det == 1.0
    assert fa <= 0.02


def test_detection_at_false_alarm_respects_budget() -> None:
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 1, 1000)
    fault = rng.normal(3, 1, 200)
    scores = np.concatenate([normal, fault])
    labels = np.concatenate([np.zeros(1000), np.ones(200)]).astype(int)
    det, thr, fa = detection_at_false_alarm(scores, labels, target_fa=0.05)
    assert 0.0 <= fa <= 0.07
    assert det > 0.5
