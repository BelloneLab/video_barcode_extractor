"""Tests for the DTW-based nonlinear time alignment."""
from __future__ import annotations

import numpy as np
import pytest

from vbe.core.alignment import (
    aligned_time_for_video_times, sanitize_align_model,
)
from vbe.core.dtw import dtw_alignment


def square_wave(period_s, fps, duration_s, t0=0.0):
    t = np.arange(0.0, duration_s, 1.0 / fps) + t0
    half = period_s / 2.0
    v = ((t - t0) % period_s < half).astype(np.float64)
    return t, v


def test_dtw_returns_model_with_anchors():
    fps = 60.0
    t_vid, v_vid = square_wave(1.0, fps, 6.0)
    t_ref = t_vid + 0.2
    model = dtw_alignment(t_vid, v_vid, t_ref, v_vid.copy(), fps=fps)
    assert model is not None
    assert model['method'] == 'dtw'
    assert model['edge_pairs'] >= 2
    assert len(model['video_edges']) == len(model['ref_edges'])
    assert model['edge_pairs'] == len(model['video_edges'])


def test_dtw_recovers_constant_shift_when_pre_aligned():
    """DTW with fixed (0,0) -> (n,m) endpoints cannot recover a pure global
    shift between non-overlapping signals (the path is forced through the
    corners). In the app the pipeline pre-aligns with cross-correlation
    first, then runs DTW on the residuals. Here we simulate that by
    pre-shifting the reference so the two signals share the same time
    support, then verify DTW returns an approximately identity map."""
    fps = 50.0
    t_vid, v_vid = square_wave(0.5, fps, 8.0)
    # Reference is the same signal on the same time grid (already aligned).
    t_ref = t_vid.copy()
    model = dtw_alignment(t_vid, v_vid, t_ref, v_vid.copy(), fps=fps,
                          max_warp_s=0.5)
    assert model is not None
    test_times = np.array([1.0, 3.0, 5.0, 7.0])
    aligned = aligned_time_for_video_times(test_times, model)
    # Identity map within DTW downsampling tolerance
    assert np.all(np.abs(aligned - test_times) < 0.15), (
        f'expected near-identity map, got {aligned - test_times}')


def test_dtw_handles_nonlinear_drift():
    """Simulate a slowly drifting clock: ref time = video time + 0.05*video time.
    The cross-correlation and linear regression methods would only see an
    average offset. DTW should at least produce a monotone increasing map."""
    fps = 50.0
    t_vid = np.arange(0.0, 10.0, 1.0 / fps)
    period_local = 0.4
    v_vid = ((t_vid % period_local) < period_local / 2.0).astype(np.float64)
    # Reference signal: same on/off pattern but with a smooth time stretch
    t_ref = t_vid + 0.05 * t_vid
    model = dtw_alignment(t_vid, v_vid, t_ref, v_vid.copy(), fps=fps,
                          max_warp_s=1.0)
    assert model is not None
    # The recovered video_edges must be strictly monotone
    va = np.asarray(model['video_edges'])
    ra = np.asarray(model['ref_edges'])
    assert np.all(np.diff(va) > 0)
    assert np.all(np.diff(ra) > 0)


def test_dtw_none_on_flat_signal():
    fps = 30.0
    t = np.linspace(0, 5, 150)
    flat = np.zeros_like(t)
    sq = ((t % 1.0) < 0.5).astype(np.float64)
    assert dtw_alignment(t, flat, t, sq, fps=fps) is None


def test_sanitize_align_model_accepts_dtw():
    model = {
        'method': 'dtw',
        'video_edges': [0.0, 1.0, 2.0, 3.0],
        'ref_edges':   [0.1, 1.05, 2.02, 3.01],
        'edge_pairs':  4,
        'slope': 1.0, 'intercept': 0.0, 'offset': 0.0,
    }
    clean = sanitize_align_model(model)
    assert clean['method'] == 'dtw'
    assert clean['edge_pairs'] == 4
