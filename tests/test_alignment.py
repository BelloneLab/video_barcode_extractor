"""Unit tests for pure-NumPy alignment utilities."""
from __future__ import annotations

import numpy as np
import pytest

from vbe.core.alignment import (
    aligned_time_for_video_times, default_align_model, edge_times,
    estimate_xcorr_offset, norm01, pair_edges, sanitize_align_model,
    video_time_from_aligned_time,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def square_wave(period_s: float, fps: float, duration_s: float,
                t0: float = 0.0):
    """Return (t, v) for a 50%-duty-cycle square wave starting at t0."""
    t = np.arange(0.0, duration_s, 1.0 / fps) + t0
    half = period_s / 2.0
    v = ((t - t0) % period_s < half).astype(np.float32)
    return t, v


# ---------------------------------------------------------------------------
# norm01
# ---------------------------------------------------------------------------
def test_norm01_constant_input_does_not_explode():
    x = np.full(10, 3.14, dtype=np.float64)
    out = norm01(x)
    assert np.all(np.isfinite(out))


def test_norm01_maps_to_unit_interval():
    x = np.array([-2.0, 0.0, 5.0, 10.0])
    out = norm01(x)
    assert out.min() == pytest.approx(0.0, abs=1e-6)
    assert out.max() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# edge_times
# ---------------------------------------------------------------------------
def test_edge_times_counts_transitions_on_square_wave():
    t, v = square_wave(period_s=1.0, fps=100.0, duration_s=5.0)
    edges, dirs = edge_times(t, v)
    # In 5 seconds with 1s period we expect ~9 transitions (rise/fall every 0.5s)
    assert 8 <= len(edges) <= 10
    assert np.all(np.isin(dirs, [-1, 1]))


def test_edge_times_alternating_directions():
    t, v = square_wave(period_s=0.5, fps=200.0, duration_s=4.0)
    edges, dirs = edge_times(t, v)
    assert len(edges) > 4
    # Consecutive edges must alternate direction
    assert np.all(np.diff(dirs) != 0)


def test_edge_times_empty_on_flat_signal():
    t = np.linspace(0, 1, 100)
    v = np.zeros_like(t)
    edges, dirs = edge_times(t, v)
    assert edges.size == 0
    assert dirs.size == 0


# ---------------------------------------------------------------------------
# pair_edges
# ---------------------------------------------------------------------------
def test_pair_edges_identical_sequences_pair_one_to_one():
    t, v = square_wave(period_s=1.0, fps=60.0, duration_s=6.0)
    edges, dirs = edge_times(t, v)
    pv, pr = pair_edges(edges, dirs, edges, dirs, offset=0.0, fps=60.0)
    assert pv.size == edges.size
    assert pr.size == edges.size
    assert np.allclose(pv, pr)


def test_pair_edges_recovers_constant_offset():
    t, v = square_wave(period_s=1.0, fps=60.0, duration_s=8.0)
    ext_e, ext_d = edge_times(t, v)
    ref_e = ext_e - 0.30   # reference is 0.30 s ahead in time
    ref_d = ext_d
    pv, pr = pair_edges(ext_e, ext_d, ref_e, ref_d, offset=0.30, fps=60.0)
    # We should pair most edges and the per-pair shift should be ~0.30
    assert pv.size >= ext_e.size - 1
    assert np.allclose(pv - pr, 0.30, atol=1e-6)


# ---------------------------------------------------------------------------
# estimate_xcorr_offset
# ---------------------------------------------------------------------------
def test_xcorr_recovers_known_shift():
    fps = 60.0
    t_vid, v_vid = square_wave(1.0, fps, 6.0)
    # Reference is the same wave shifted by +0.4 s in its own clock
    t_ref = t_vid + 0.4
    v_ref = v_vid.copy()
    result = estimate_xcorr_offset(
        t_vid, v_vid, t_ref, v_ref,
        max_lag_s=1.0, fps=fps)
    assert result is not None
    # estimate_xcorr_offset returns the offset such that aligned = video - offset
    # matches the reference. So we expect offset ~ -0.4.
    assert abs(result['offset'] - (-0.4)) < 0.05
    assert result['peak'] > 0.9


def test_xcorr_returns_none_on_flat_signal():
    t = np.linspace(0, 5, 300)
    flat = np.zeros_like(t)
    sq = ((t % 1.0) < 0.5).astype(np.float32)
    assert estimate_xcorr_offset(t, flat, t, sq, max_lag_s=1.0, fps=60.0) is None


# ---------------------------------------------------------------------------
# aligned_time round-trip
# ---------------------------------------------------------------------------
def test_aligned_time_roundtrip_cross_correlation():
    model = default_align_model(offset=0.0)
    times = np.linspace(0, 10, 200)
    aligned = aligned_time_for_video_times(times, model, offset_fallback=0.7)
    back = np.array([video_time_from_aligned_time(a, model, offset_fallback=0.7)
                     for a in aligned])
    assert np.allclose(times, back, atol=1e-9)


def test_aligned_time_linear_regression_uses_slope_intercept():
    model = {
        'method': 'linear_regression',
        'slope': 1.0001,
        'intercept': -0.5,
        'video_edges': [0.0, 5.0, 10.0],
        'ref_edges': [-0.5, 4.5005, 9.501],
        'edge_pairs': 3,
        'offset': 0.0,
    }
    times = np.array([0.0, 5.0, 10.0])
    aligned = aligned_time_for_video_times(times, model)
    expected = 1.0001 * times - 0.5
    assert np.allclose(aligned, expected, atol=1e-9)


def test_aligned_time_edge_interpolation():
    model = {
        'method': 'edge_interpolation',
        'video_edges': [0.0, 5.0, 10.0],
        'ref_edges': [0.1, 5.2, 10.1],
        'edge_pairs': 3,
        'offset': 0.0,
        'slope': 1.0,
        'intercept': 0.0,
    }
    # At the knots the map should exactly reproduce the reference times
    out = aligned_time_for_video_times(np.array(model['video_edges']), model)
    assert np.allclose(out, model['ref_edges'], atol=1e-9)


# ---------------------------------------------------------------------------
# sanitize_align_model: regression for the Pass-A bug
# ---------------------------------------------------------------------------
def test_sanitize_align_model_recovers_from_scalar_ref_edges():
    # Pre-Pass-A, _auto_align would overwrite the 'ref_edges' list with an int.
    corrupted = {
        'method': 'linear_regression',
        'video_edges': [0.0, 1.0, 2.0],
        'ref_edges': 5,                # corrupted: was a count, not a list
        'edge_pairs': 3,
        'slope': 1.0,
        'intercept': 0.0,
        'offset': 0.0,
    }
    clean = sanitize_align_model(corrupted)
    # video_edges and ref_edges must be lists of equal length; method must
    # downgrade gracefully because the corrupted data offers <2 valid pairs.
    assert isinstance(clean['video_edges'], list)
    assert isinstance(clean['ref_edges'], list)
    assert len(clean['video_edges']) == len(clean['ref_edges'])
    assert clean['edge_pairs'] == len(clean['video_edges'])
    if clean['edge_pairs'] < 2:
        assert clean['method'] == 'cross_correlation'


def test_sanitize_align_model_passes_through_clean_input():
    src = {
        'method': 'edge_interpolation',
        'video_edges': [0.0, 1.0, 2.0],
        'ref_edges': [0.1, 1.05, 2.02],
        'edge_pairs': 3,
        'slope': 1.0,
        'intercept': 0.0,
        'offset': 0.0,
    }
    clean = sanitize_align_model(src)
    assert clean['method'] == 'edge_interpolation'
    assert clean['edge_pairs'] == 3
    assert clean['video_edges'] == [0.0, 1.0, 2.0]
    assert clean['ref_edges'] == [0.1, 1.05, 2.02]
