"""Unit tests for signal-quality diagnostics."""
from __future__ import annotations

import math

import numpy as np
import pytest

from vbe.core.quality import (
    edge_interval_stats, saturated_fraction, signal_snr_db,
)


def test_snr_is_high_for_clean_square_wave():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 10, 1000)
    sq = ((t % 1.0) < 0.5).astype(np.float64) * 200.0
    noisy = sq + rng.normal(0, 2, sq.shape)
    snr = signal_snr_db(noisy, threshold=100.0)
    assert snr > 25.0  # ~200 / ~2 ~ 100, 20*log10(100) = 40 dB


def test_snr_nan_on_too_few_off_samples():
    v = np.full(10, 200.0)
    assert math.isnan(signal_snr_db(v, threshold=100.0))


def test_edge_interval_stats_on_uniform_pulses():
    fps = 100.0
    t = np.arange(0, 10, 1.0 / fps)
    period = 1.0
    v = ((t % period) < period / 2.0).astype(np.float64)
    stats = edge_interval_stats(t, v)
    assert stats['n_edges'] >= 18
    assert abs(stats['mean_s'] - 0.5) < 0.05
    assert stats['cv'] < 0.05  # uniform pulses have tiny CV


def test_edge_interval_stats_nan_when_too_few_edges():
    t = np.linspace(0, 1, 100)
    v = np.zeros_like(t)
    stats = edge_interval_stats(t, v)
    assert stats['n_edges'] == 0
    assert math.isnan(stats['mean_s'])


def test_saturated_fraction_zero_for_continuous_signal():
    t = np.linspace(0, 10, 1000)
    v = np.sin(t)
    assert saturated_fraction(v) < 0.05


def test_saturated_fraction_high_for_binary_signal():
    v = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1], dtype=np.float64)
    assert saturated_fraction(v) == 1.0
