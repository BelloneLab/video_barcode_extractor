"""Pure-NumPy alignment utilities: edges, pairing, cross-correlation, time-maps.

Three alignment modes are supported:

1. cross_correlation
   One global time offset estimated from the lag of maximum normalized
   correlation between the (binarized) video signal and the reference trace.

2. linear_regression
   Pair video edges to reference edges, then least-squares fit
   ref_t = slope * video_t + intercept. Captures linear clock skew.

3. edge_interpolation
   Piecewise-linear time map through the matched edges. Captures drift that
   is not strictly linear (e.g. dropped frames, thermal drift).

All functions are stateless and Qt-free.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

# Pearson r below this value flags an alignment as unreliable. The app warns
# the user instead of refusing outright.
DEFAULT_MIN_PEAK_R = 0.30


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo + 1e-9)


# ---------------------------------------------------------------------------
# Edge detection
# ---------------------------------------------------------------------------
def edge_times(t: np.ndarray, v: np.ndarray, threshold_norm: float = 0.5):
    """Return sub-frame edge times and signed directions of every transition.

    A "state" is the binarization `norm01(v) >= threshold_norm`. For each
    pair of adjacent samples that bracket a crossing, we estimate the true
    crossing time by linear interpolation across the rising/falling edge:

        alpha = (thr - v[i-1]) / (v[i] - v[i-1])
        edge_t = t[i-1] + alpha * (t[i] - t[i-1])

    At 30 fps the previous midpoint formula bounded edge resolution to half
    a frame (~17 ms). Linear interpolation across a smoothed step pushes that
    below 1 frame, often to a few hundred microseconds depending on signal
    sharpness. `edge_dir` is +1 for off-to-on and -1 for on-to-off.
    """
    t = np.asarray(t, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    mask = np.isfinite(t) & np.isfinite(v)
    t, v = t[mask], v[mask]
    if len(t) < 2:
        return np.array([], dtype=np.float64), np.array([], dtype=np.int8)
    order = np.argsort(t)
    t, v = t[order], v[order]
    v_norm = norm01(v)
    state = v_norm >= threshold_norm
    changes = np.flatnonzero(np.diff(state.astype(np.int8)) != 0) + 1
    if len(changes) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.int8)
    v_lo = v_norm[changes - 1]
    v_hi = v_norm[changes]
    denom = v_hi - v_lo
    # Defensive: if the two bracketing samples have the same normalized
    # value (rare; numerical floor), fall back to the midpoint.
    alpha = np.where(np.abs(denom) > 1e-12,
                     (threshold_norm - v_lo) / denom,
                     0.5)
    alpha = np.clip(alpha, 0.0, 1.0)
    edge_t = t[changes - 1] + alpha * (t[changes] - t[changes - 1])
    edge_dir = np.where(state[changes], 1, -1).astype(np.int8)
    return edge_t, edge_dir


def pair_edges(video_edges, video_dirs, ref_edges, ref_dirs,
               offset: float, fps: float):
    """Match each video edge to the nearest reference edge.

    Tolerance is `max(3 / fps, min(1.0, 0.45 * typical_spacing))`. If almost
    no pairs are found but the edge counts are close, fall back to ordinal
    pairing.
    """
    video_edges = np.asarray(video_edges, dtype=np.float64)
    ref_edges = np.asarray(ref_edges, dtype=np.float64)
    if len(video_edges) == 0 or len(ref_edges) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    ref_on_video = ref_edges + offset
    spacings = []
    for arr in (video_edges, ref_on_video):
        if len(arr) > 1:
            d = np.diff(arr)
            spacings.extend(d[d > 0].tolist())
    typical = float(np.median(spacings)) if spacings else 1.0
    tol = max(3.0 / max(fps, 1e-9), min(1.0, typical * 0.45))

    pairs_v, pairs_r = [], []
    j = 0
    for ve, vd in zip(video_edges, video_dirs):
        while j < len(ref_on_video) and ref_on_video[j] < ve - tol:
            j += 1
        candidates = []
        for k in (j, j + 1):
            if k < len(ref_on_video) and abs(ref_on_video[k] - ve) <= tol:
                if len(ref_dirs) <= k or ref_dirs[k] == vd:
                    candidates.append(k)
        if candidates:
            k = min(candidates, key=lambda ii: abs(ref_on_video[ii] - ve))
            pairs_v.append(float(ve))
            pairs_r.append(float(ref_edges[k]))
            j = k + 1

    if len(pairs_v) < 2:
        diff = abs(len(video_edges) - len(ref_edges))
        slack = max(2, int(0.15 * min(len(video_edges), len(ref_edges))))
        if diff <= slack:
            n = min(len(video_edges), len(ref_edges))
            pairs_v = video_edges[:n].astype(float).tolist()
            pairs_r = ref_edges[:n].astype(float).tolist()

    return (np.asarray(pairs_v, dtype=np.float64),
            np.asarray(pairs_r, dtype=np.float64))


# ---------------------------------------------------------------------------
# Cross-correlation
# ---------------------------------------------------------------------------
def estimate_xcorr_offset(sig_t, output, t_csv, v_csv,
                          max_lag_s: float, fps: float,
                          view_range=None):
    """FFT-based normalized cross-correlation between `output` and `v_csv`.

    Both signals are resampled onto a common uniform grid at the video's
    median sample interval. The mean is removed; the cross-correlation is
    computed via `scipy.signal.fftconvolve` (O(N log N)) and normalized to
    give Pearson r at each lag, exact in the limit where the lag is small
    compared to the overlapping signal length.

    `view_range` restricts the evaluation window in video time. If the window
    is too small or the signal is flat, the full signal is used.

    Returns a dict (offset, lag, peak, lags, corr_norm, window_s,
    min_overlap_samples), or None if alignment cannot be estimated. The
    sign convention is: `aligned_time = video_time - offset` should overlay
    the video signal on the reference.
    """
    t_csv = np.asarray(t_csv, dtype=np.float64)
    v_csv = np.asarray(v_csv, dtype=np.float64)
    valid = np.isfinite(t_csv) & np.isfinite(v_csv)
    if valid.sum() < 2:
        return None
    order = np.argsort(t_csv[valid])
    t_csv = t_csv[valid][order]
    v_csv = v_csv[valid][order]

    sig_t = np.asarray(sig_t, dtype=np.float64)
    out = np.asarray(output, dtype=np.float64)
    finite_sig = np.isfinite(sig_t) & np.isfinite(out)
    sig_t, out = sig_t[finite_sig], out[finite_sig]
    if len(sig_t) < 20:
        return None

    if view_range is not None:
        x0, x1 = float(view_range[0]), float(view_range[1])
        if (x1 - x0) < (3.0 / max(fps, 1e-9)) or np.nanstd(out[(sig_t >= x0) & (sig_t <= x1)] if ((sig_t >= x0) & (sig_t <= x1)).sum() > 0 else out) < 1e-9:
            x0, x1 = float(sig_t[0]), float(sig_t[-1])
    else:
        x0, x1 = float(sig_t[0]), float(sig_t[-1])

    dt = float(np.nanmedian(np.diff(sig_t))) if len(sig_t) > 1 else 1.0 / max(fps, 1e-9)
    if not np.isfinite(dt) or dt <= 0:
        dt = 1.0 / max(fps, 1e-9)

    # Common grid: intersection of usable times, padded by max_lag on each side
    t_lo = max(x0, float(t_csv[0]) - max_lag_s)
    t_hi = min(x1, float(t_csv[-1]) + max_lag_s)
    if (t_hi - t_lo) < 1.0:
        return None
    grid = np.arange(t_lo, t_hi + 0.5 * dt, dt, dtype=np.float64)
    if len(grid) < 20:
        return None
    a = np.interp(grid, sig_t, out, left=np.nan, right=np.nan)
    b = np.interp(grid, t_csv, v_csv, left=np.nan, right=np.nan)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 20:
        return None
    a = a[mask]
    b = b[mask]
    grid = grid[mask]

    a_c = a - np.mean(a)
    b_c = b - np.mean(b)
    a_std = float(np.std(a_c))
    b_std = float(np.std(b_c))
    if a_std < 1e-9 or b_std < 1e-9:
        return None

    n = len(a_c)
    c = fftconvolve(a_c, b_c[::-1], mode='full')
    lags_samples = np.arange(-(n - 1), n)
    lags_s = lags_samples * dt
    denom = a_std * b_std * n
    r_vals = c / max(denom, 1e-12)

    keep = np.abs(lags_s) <= max_lag_s
    if not keep.any():
        return None
    r_vals = r_vals[keep]
    lags_s = lags_s[keep]

    best_idx = int(np.nanargmax(r_vals))
    best_lag = float(lags_s[best_idx])
    # scipy/fftconvolve convention: at this lag, a[l] ≈ b[l - lag/dt].
    # For our app `aligned_t = sig_t - offset` to overlay the video signal on
    # the reference, we need `offset = best_lag` (no sign flip).
    best_offset = best_lag

    return {
        'offset': best_offset,
        'lag': best_lag,
        'peak': float(r_vals[best_idx]),
        'lags': lags_s,
        'corr_norm': r_vals,
        'window_s': (float(grid[0]), float(grid[-1])),
        'min_overlap_samples': int(n),
    }


def ransac_regression(pairs_v, pairs_r, residual_threshold_s: float = 0.05,
                      max_trials: int = 100):
    """Robust line fit y = slope * x + intercept on matched edge pairs.

    Returns a dict {slope, intercept, inliers, total, inlier_mask}, or None
    if fewer than 3 points are provided or RANSAC fails. The residual threshold
    is in seconds.
    """
    from skimage.measure import LineModelND, ransac

    pairs_v = np.asarray(pairs_v, dtype=np.float64).reshape(-1)
    pairs_r = np.asarray(pairs_r, dtype=np.float64).reshape(-1)
    if pairs_v.size < 3 or pairs_v.size != pairs_r.size:
        return None
    data = np.column_stack([pairs_v, pairs_r])
    try:
        model, inliers = ransac(
            data, LineModelND,
            min_samples=2,
            residual_threshold=residual_threshold_s,
            max_trials=max_trials)
    except Exception:
        return None
    if model is None or inliers is None:
        return None
    origin, direction = model.params
    if abs(direction[0]) < 1e-12:
        return None
    slope = float(direction[1] / direction[0])
    intercept = float(origin[1] - slope * origin[0])
    return {
        'slope': slope,
        'intercept': intercept,
        'inliers': int(inliers.sum()),
        'total': int(pairs_v.size),
        'inlier_mask': inliers,
    }


# ---------------------------------------------------------------------------
# Time mapping
# ---------------------------------------------------------------------------
def aligned_time_for_video_times(times, align_model, offset_fallback: float = 0.0):
    times = np.asarray(times, dtype=np.float64)
    method = align_model.get('method', 'cross_correlation')
    if method == 'linear_regression' and align_model.get('edge_pairs', 0) >= 2:
        return (float(align_model.get('slope', 1.0)) * times
                + float(align_model.get('intercept', 0.0)))
    if method in ('edge_interpolation', 'dtw') and align_model.get('edge_pairs', 0) >= 2:
        x = np.asarray(align_model.get('video_edges', []), dtype=np.float64).reshape(-1)
        y = np.asarray(align_model.get('ref_edges', []), dtype=np.float64).reshape(-1)
        if x.size >= 2 and y.size == x.size:
            out = np.interp(times, x, y)
            left = times < x[0]
            right = times > x[-1]
            if np.any(left):
                slope = (y[1] - y[0]) / (x[1] - x[0] + 1e-12)
                out[left] = y[0] + slope * (times[left] - x[0])
            if np.any(right):
                slope = (y[-1] - y[-2]) / (x[-1] - x[-2] + 1e-12)
                out[right] = y[-1] + slope * (times[right] - x[-1])
            return out
    return times - float(offset_fallback)


def video_time_from_aligned_time(aligned_t, align_model, offset_fallback: float = 0.0):
    method = align_model.get('method', 'cross_correlation')
    if method == 'linear_regression' and align_model.get('edge_pairs', 0) >= 2:
        slope = float(align_model.get('slope', 1.0))
        return ((aligned_t - float(align_model.get('intercept', 0.0)))
                / (slope if abs(slope) > 1e-12 else 1.0))
    if method in ('edge_interpolation', 'dtw') and align_model.get('edge_pairs', 0) >= 2:
        x = np.asarray(align_model.get('video_edges', []), dtype=np.float64).reshape(-1)
        y = np.asarray(align_model.get('ref_edges', []), dtype=np.float64).reshape(-1)
        if x.size >= 2 and y.size == x.size:
            return float(np.interp(aligned_t, y, x, left=x[0], right=x[-1]))
    return aligned_t + float(offset_fallback)


def alignment_metrics(output_norm, aligned_t, t_csv, v_csv, fps,
                      x_range=None):
    csv_norm = norm01(v_csv).astype(np.float64)
    v_interp = np.interp(aligned_t, t_csv, csv_norm,
                         left=np.nan, right=np.nan)
    mask = ~np.isnan(v_interp)
    if x_range is not None:
        x0, x1 = x_range
        lo, hi = min(x0, x1), max(x0, x1)
        mask &= (aligned_t >= lo) & (aligned_t <= hi)
    if mask.sum() <= 10:
        return None
    a = output_norm[mask].astype(np.float64)
    b = v_interp[mask].astype(np.float64)
    r = float(np.corrcoef(a, b)[0, 1])
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    mae = float(np.mean(np.abs(a - b)))
    agree = float(np.mean((a >= 0.5) == (b >= 0.5)))
    overlap_s = float(mask.sum() / max(fps, 1e-9))
    return {
        'r': r,
        'rmse': rmse,
        'mae': mae,
        'agree': agree,
        'overlap_s': overlap_s,
        'samples': int(mask.sum()),
    }


# ---------------------------------------------------------------------------
# Model dict shape (kept as dict for project-file backward compatibility)
# ---------------------------------------------------------------------------
def default_align_model(offset: float = 0.0) -> dict:
    return {
        'method': 'cross_correlation',
        'offset': float(offset),
        'slope': 1.0,
        'intercept': 0.0,
        'video_edges': [],
        'ref_edges': [],
        'edge_pairs': 0,
    }


def _as_1d_float_list(value):
    if value is None:
        return []
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return []
    arr = arr[np.isfinite(arr)]
    return arr.astype(float).tolist()


def sanitize_align_model(model, default_offset: float = 0.0) -> dict:
    if not isinstance(model, dict):
        return default_align_model(default_offset)
    clean = default_align_model(default_offset)
    clean.update(model)
    clean['video_edges'] = _as_1d_float_list(clean.get('video_edges'))
    clean['ref_edges'] = _as_1d_float_list(clean.get('ref_edges'))
    n = min(len(clean['video_edges']), len(clean['ref_edges']))
    clean['video_edges'] = clean['video_edges'][:n]
    clean['ref_edges'] = clean['ref_edges'][:n]
    try:
        clean['edge_pairs'] = min(int(clean.get('edge_pairs', n) or 0), n)
    except Exception:
        clean['edge_pairs'] = n
    if clean['edge_pairs'] < 2 and clean.get('method') in (
            'linear_regression', 'edge_interpolation', 'dtw'):
        clean['method'] = 'cross_correlation'
    for key, default in [('offset', default_offset), ('slope', 1.0), ('intercept', 0.0)]:
        try:
            clean[key] = float(clean.get(key, default))
        except Exception:
            clean[key] = default
    return clean
