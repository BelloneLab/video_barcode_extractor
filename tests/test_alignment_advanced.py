"""Additional tests for the FFT cross-correlation and RANSAC regression."""
from __future__ import annotations

import numpy as np
import pytest

from vbe.core.alignment import (
    estimate_xcorr_offset, ransac_regression,
)


def square_wave(period_s: float, fps: float, duration_s: float, t0: float = 0.0):
    t = np.arange(0.0, duration_s, 1.0 / fps) + t0
    half = period_s / 2.0
    v = ((t - t0) % period_s < half).astype(np.float64)
    return t, v


def test_fft_xcorr_recovers_known_shift_short_signal():
    fps = 60.0
    t_vid, v_vid = square_wave(1.0, fps, 6.0)
    t_ref = t_vid + 0.4
    v_ref = v_vid.copy()
    r = estimate_xcorr_offset(t_vid, v_vid, t_ref, v_ref,
                              max_lag_s=1.0, fps=fps)
    assert r is not None
    assert abs(r['offset'] - (-0.4)) < 0.05
    assert r['peak'] > 0.9


def test_fft_xcorr_handles_long_signal_quickly():
    """O(N log N) should remain fast on a 60-second signal at 100 fps."""
    import time
    fps = 100.0
    duration = 60.0
    t_vid, v_vid = square_wave(0.5, fps, duration)
    t_ref = t_vid + 0.25
    v_ref = v_vid.copy()
    t0 = time.perf_counter()
    r = estimate_xcorr_offset(t_vid, v_vid, t_ref, v_ref,
                              max_lag_s=2.0, fps=fps)
    elapsed = time.perf_counter() - t0
    assert r is not None
    assert abs(r['offset'] - (-0.25)) < 0.05
    # Generous bound: should run well under a second even on slow CI
    assert elapsed < 1.0, f'fft xcorr too slow: {elapsed:.2f}s'


def test_fft_xcorr_returns_none_on_flat_signal():
    t = np.linspace(0, 5, 300)
    flat = np.zeros_like(t)
    sq = ((t % 1.0) < 0.5).astype(np.float64)
    assert estimate_xcorr_offset(t, flat, t, sq, max_lag_s=1.0, fps=60.0) is None


def test_ransac_recovers_slope_under_outliers():
    rng = np.random.default_rng(42)
    n = 50
    x = np.linspace(0, 100, n)
    y = 1.001 * x - 0.20 + rng.normal(0, 0.001, n)
    # Corrupt 20% of the points with large outliers
    bad = rng.choice(n, size=10, replace=False)
    y[bad] += rng.normal(0, 5.0, 10)
    result = ransac_regression(x, y, residual_threshold_s=0.05)
    assert result is not None
    assert abs(result['slope'] - 1.001) < 1e-2
    assert abs(result['intercept'] - (-0.20)) < 0.5
    assert result['inliers'] >= int(0.7 * n)


def test_ransac_returns_none_for_too_few_points():
    assert ransac_regression([0, 1], [0, 1]) is None
