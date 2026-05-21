"""Tests for the sub-frame edge timing in edge_times."""
from __future__ import annotations

import numpy as np

from vbe.core.alignment import edge_times


def test_subframe_recovers_true_crossing_for_linear_ramp():
    """A linear ramp with a symmetric range crosses normalized 0.5 at t=0.5.
    Sub-frame interpolation should recover this to numerical precision."""
    fps = 100.0
    # Use np.linspace with endpoint=True so v has a symmetric range [-1, +1]
    # and normalized 0.5 corresponds to v=0 at t=0.5 exactly.
    t = np.linspace(0.0, 1.0, int(fps) + 1)
    v = (t - 0.5) * 2.0  # ranges from -1.0 to +1.0 symmetrically
    edges, dirs = edge_times(t, v)
    assert len(edges) == 1
    assert dirs[0] == 1
    # Midpoint formula would give 0.5 ± 0.005 s. Sub-frame must be much tighter.
    assert abs(edges[0] - 0.5) < 1e-6


def test_subframe_is_strictly_better_than_midpoint_on_offgrid_step():
    """A square wave whose true edge falls between two samples should be
    estimated closer than half a frame."""
    fps = 30.0
    dt = 1.0 / fps
    # Edge at t = 1.0 + 0.4*dt (between samples 30 and 31)
    true_edge = 1.0 + 0.4 * dt
    t = np.arange(0.0, 2.0, dt)
    # Smoothed step: use logistic so two samples bracket the threshold
    # asymmetrically, which is what sub-frame timing should exploit.
    v = 1.0 / (1.0 + np.exp(-30.0 * (t - true_edge)))
    edges, _ = edge_times(t, v)
    assert len(edges) == 1
    midpoint_estimate = (t[t < true_edge][-1] + t[t > true_edge][0]) / 2.0
    err_subframe = abs(edges[0] - true_edge)
    err_midpoint = abs(midpoint_estimate - true_edge)
    assert err_subframe < err_midpoint, (
        f'sub-frame error {err_subframe} should beat midpoint {err_midpoint}')
    # The sub-frame error should be a small fraction of a frame
    assert err_subframe < 0.5 * dt


def test_edge_directions_still_alternate():
    """The directional sign should still be +1, -1, +1, -1, ... for a square wave."""
    fps = 100.0
    t = np.arange(0.0, 4.0, 1.0 / fps)
    period = 1.0
    v = ((t % period) < period / 2.0).astype(np.float64)
    edges, dirs = edge_times(t, v)
    assert len(edges) >= 6
    # All consecutive direction differences must be nonzero (no two same in a row)
    assert np.all(np.diff(dirs) != 0)
