"""Unit tests for the thresholding methods."""
from __future__ import annotations

import numpy as np
import pytest

from vbe.core.thresholds import (
    auto_threshold, threshold_kapur, threshold_mad,
    threshold_otsu, threshold_percentile, threshold_triangle,
)


def _bimodal_balanced(seed: int = 0):
    rng = np.random.default_rng(seed)
    off = rng.normal(20, 3, 500)
    on = rng.normal(180, 5, 500)
    labels = np.concatenate([np.zeros(500), np.ones(500)])
    return np.concatenate([off, on]).astype(np.float64), labels


def _bimodal_sparse(seed: int = 0):
    rng = np.random.default_rng(seed)
    off = rng.normal(20, 3, 1900)
    on = rng.normal(180, 5, 100)   # 5% on, sparse pulses
    labels = np.concatenate([np.zeros(1900), np.ones(100)])
    return np.concatenate([off, on]).astype(np.float64), labels


def _classification_accuracy(values, labels, threshold):
    predicted = values > threshold
    return float(np.mean(predicted == labels.astype(bool)))


def test_all_methods_return_value_in_signal_range():
    x, _ = _bimodal_balanced()
    lo, hi = x.min(), x.max()
    for fn in (threshold_otsu, threshold_triangle, threshold_kapur,
               threshold_mad, threshold_percentile):
        t = fn(x)
        assert lo <= t <= hi, f'{fn.__name__} returned {t} outside [{lo}, {hi}]'


def test_otsu_separates_balanced_bimodal():
    x, labels = _bimodal_balanced()
    t = threshold_otsu(x)
    assert _classification_accuracy(x, labels, t) > 0.99


def test_triangle_separates_sparse_bimodal():
    """For a 5%-duty sparse signal, Triangle should classify on-samples correctly
    while Otsu typically picks a threshold that misclassifies many off-samples
    or merges the small on-cluster into the off-cluster.
    """
    x, labels = _bimodal_sparse()
    acc_tri = _classification_accuracy(x, labels, threshold_triangle(x))
    assert acc_tri > 0.98


def test_auto_picks_triangle_for_sparse():
    x, _ = _bimodal_sparse()
    _, method = auto_threshold(x)
    assert method == 'Triangle'


def test_auto_picks_otsu_for_balanced():
    x, _ = _bimodal_balanced()
    _, method = auto_threshold(x)
    assert method == 'Otsu'


def test_mad_threshold_robust_to_outliers():
    x = np.concatenate([np.zeros(1000), [1e6]])   # one huge outlier
    t = threshold_mad(x, k=3)
    # MAD is essentially zero for a near-constant series, so threshold is small
    assert t < 100.0


def test_percentile_returns_correct_quantile():
    x = np.arange(101, dtype=np.float64)
    t = threshold_percentile(x, p=90.0)
    assert abs(t - 90.0) < 1.0
