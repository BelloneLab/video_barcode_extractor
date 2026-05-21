"""Dynamic Time Warping alignment.

DTW finds a monotone, locally-flexible time map between two signals by
optimizing the cumulative cost of a warping path through a 2D cost matrix
under three permitted steps (diagonal, vertical, horizontal). Unlike
cross-correlation (one global offset) or edge regression (a line through
matched landmarks), DTW can capture non-linear drift, transient clock
skew, and dropped frames between the video signal and the reference.

Implementation notes
--------------------
We resample both signals onto a common uniform grid at the video's median
sample interval, normalize each to [0, 1] (so amplitude differences do not
dominate), and downsample to at most `max_samples` (default 600) points to
keep DTW tractable on long signals. The Sakoe-Chiba band limits the search
to |i - j| <= W where W is derived from `max_warp_s`.

Returned anchors are time pairs (video_time, reference_time) along the
optimal path. We sub-sample these to at most `n_anchors` points so the
saved model stays small. The app reuses the same `aligned_time_for_video_times`
path that `edge_interpolation` uses to evaluate the time map at arbitrary
video times.
"""
from __future__ import annotations

import numpy as np


def _resample(t, v, grid):
    return np.interp(grid, t, v, left=np.nan, right=np.nan)


def _norm01_safe(x):
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    return (x - lo) / (hi - lo + 1e-9)


def _dtw_path(a: np.ndarray, b: np.ndarray, window: int):
    """Compute the DTW warping path with Sakoe-Chiba band.

    Returns (path_array, normalized_cost). path_array has shape (K, 2) with
    integer indices into a and b. Cost is the cumulative L1 distance along
    the path, divided by the path length.
    """
    n = int(len(a))
    m = int(len(b))
    window = max(window, abs(n - m))
    INF = np.inf
    D = np.full((n + 1, m + 1), INF, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        j_lo = max(1, i - window)
        j_hi = min(m, i + window)
        a_i = a[i - 1]
        # vectorized cost row for the band
        b_slice = b[j_lo - 1:j_hi]
        dists = np.abs(a_i - b_slice)
        # vectorize prev-row mins
        prev_diag = D[i - 1, j_lo - 1:j_hi]
        prev_up = D[i - 1, j_lo:j_hi + 1]
        # iterative because D[i, j-1] is same-row
        row = D[i]
        for k, j in enumerate(range(j_lo, j_hi + 1)):
            left = row[j - 1]
            best = prev_diag[k]
            if prev_up[k] < best:
                best = prev_up[k]
            if left < best:
                best = left
            row[j] = dists[k] + best
    # Backtrack
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        if i == 1 and j == 1:
            break
        c_diag = D[i - 1, j - 1] if i > 0 and j > 0 else INF
        c_up   = D[i - 1, j]     if i > 0 else INF
        c_left = D[i, j - 1]     if j > 0 else INF
        if c_diag <= c_up and c_diag <= c_left:
            i, j = i - 1, j - 1
        elif c_up <= c_left:
            i = i - 1
        else:
            j = j - 1
    path.reverse()
    return np.asarray(path, dtype=np.int64), float(D[n, m] / max(len(path), 1))


def dtw_alignment(sig_t, sig_v, t_ref, v_ref, fps: float,
                  max_warp_s: float = 2.0,
                  max_samples: int = 600,
                  n_anchors: int = 200):
    """DTW-based time map from video signal to reference.

    Returns dict with keys:
      - 'method': 'dtw'
      - 'video_edges': list of video times (anchors)
      - 'ref_edges':   list of corresponding reference times (anchors)
      - 'edge_pairs':  int, number of anchors stored
      - 'dtw_cost':    float, normalized DTW path cost (lower = better)
      - 'dtw_window':  int, Sakoe-Chiba band width in resampled samples
      - 'downsampled_n': int, length of the resampled signal used for DTW
      - 'offset':      float, residual offset placeholder (0.0)
      - 'slope':       1.0
      - 'intercept':   0.0

    Returns None if the inputs are too short or numerically degenerate.

    The key 'video_edges' / 'ref_edges' is intentionally reused so the app's
    existing `aligned_time_for_video_times` piecewise-linear path applies
    without modification when method='dtw'.
    """
    sig_t = np.asarray(sig_t, dtype=np.float64)
    sig_v = np.asarray(sig_v, dtype=np.float64)
    t_ref = np.asarray(t_ref, dtype=np.float64)
    v_ref = np.asarray(v_ref, dtype=np.float64)
    fin_a = np.isfinite(sig_t) & np.isfinite(sig_v)
    fin_b = np.isfinite(t_ref) & np.isfinite(v_ref)
    sig_t, sig_v = sig_t[fin_a], sig_v[fin_a]
    t_ref, v_ref = t_ref[fin_b], v_ref[fin_b]
    if len(sig_t) < 10 or len(t_ref) < 10:
        return None
    if np.nanstd(sig_v) < 1e-9 or np.nanstd(v_ref) < 1e-9:
        return None

    order_a = np.argsort(sig_t)
    order_b = np.argsort(t_ref)
    sig_t, sig_v = sig_t[order_a], sig_v[order_a]
    t_ref, v_ref = t_ref[order_b], v_ref[order_b]

    dt = float(np.nanmedian(np.diff(sig_t))) if len(sig_t) > 1 else 1.0 / max(fps, 1e-9)
    if not np.isfinite(dt) or dt <= 0:
        dt = 1.0 / max(fps, 1e-9)

    # Common time interval (allowing each signal to have some unique tails)
    t0 = max(sig_t[0], t_ref[0])
    t1 = min(sig_t[-1], t_ref[-1])
    if (t1 - t0) < 5 * dt:
        return None
    grid = np.arange(t0, t1 + 0.5 * dt, dt)
    a = _resample(sig_t, sig_v, grid)
    b = _resample(t_ref, v_ref, grid)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return None
    a, b, grid = a[mask], b[mask], grid[mask]

    # Downsample to keep DTW tractable
    n = len(grid)
    if n > max_samples:
        idx = np.linspace(0, n - 1, max_samples).astype(np.int64)
        a, b, grid = a[idx], b[idx], grid[idx]
        n = len(grid)
        ds_dt = (grid[-1] - grid[0]) / max(n - 1, 1)
    else:
        ds_dt = dt

    a = _norm01_safe(a)
    b = _norm01_safe(b)

    window = max(2, int(round(max_warp_s / ds_dt)))
    window = min(window, n)
    path, cost = _dtw_path(a, b, window)
    if path.size == 0:
        return None

    video_anchors = grid[path[:, 0]]
    ref_anchors = grid[path[:, 1]]

    # Sparse anchor subset for storage
    K = len(video_anchors)
    if K > n_anchors:
        sel = np.linspace(0, K - 1, n_anchors).astype(np.int64)
        video_anchors = video_anchors[sel]
        ref_anchors = ref_anchors[sel]

    # Enforce strict monotonicity so the time map is invertible at evaluation
    # time. Tiny numerical ties get a small forward bump.
    eps = 1e-9
    for arr in (video_anchors, ref_anchors):
        for i in range(1, len(arr)):
            if arr[i] <= arr[i - 1]:
                arr[i] = arr[i - 1] + eps

    return {
        'method': 'dtw',
        'video_edges': video_anchors.astype(float).tolist(),
        'ref_edges':   ref_anchors.astype(float).tolist(),
        'edge_pairs':  int(len(video_anchors)),
        'dtw_cost':    float(cost),
        'dtw_window':  int(window),
        'downsampled_n': int(n),
        'offset':      0.0,
        'slope':       1.0,
        'intercept':   0.0,
    }
