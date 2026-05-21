"""Signal quality diagnostics for extracted ROI traces.

Three numbers tell most of the story:

1. SNR in dB. Peak-to-peak amplitude over the noise floor measured from the
   off-state (samples below threshold). High SNR means a clean step signal.
2. Edge-interval coefficient of variation (CV). Narrow CV means the inter-edge
   intervals are repeatable, which is what a barcode train looks like.
3. Saturation fraction. Fraction of samples pinned at the dynamic-range floor
   or ceiling. High saturation can hide structure inside the on-state.
"""
from __future__ import annotations

import numpy as np

from vbe.core.alignment import edge_times


def signal_snr_db(values, threshold: float) -> float:
    """Peak-to-peak / noise floor (MAD of off-state samples) expressed in dB."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 5:
        return float('nan')
    off = v[v < threshold]
    if off.size < 5:
        return float('nan')
    med_off = float(np.median(off))
    noise = float(np.median(np.abs(off - med_off)) * 1.4826)
    if noise <= 1e-12:
        return float('inf')
    amp = float(np.max(v) - np.min(v))
    if amp <= 0:
        return float('nan')
    return float(20.0 * np.log10(amp / noise))


def edge_interval_stats(times, values) -> dict:
    """Return n_edges, mean inter-edge interval, and CV (std / mean)."""
    edges, _ = edge_times(np.asarray(times), np.asarray(values))
    n = int(edges.size)
    if n < 3:
        return {'n_edges': n, 'mean_s': float('nan'), 'cv': float('nan')}
    intervals = np.diff(edges)
    m = float(np.mean(intervals))
    s = float(np.std(intervals))
    cv = float(s / m) if m > 0 else float('inf')
    return {'n_edges': n, 'mean_s': m, 'cv': cv}


def saturated_fraction(values, rel_eps: float = 1e-3) -> float:
    """Fraction of samples within rel_eps of the min or max."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    lo, hi = float(v.min()), float(v.max())
    rng = hi - lo
    if rng <= 0:
        return 1.0
    sat = np.sum((v <= lo + rel_eps * rng) | (v >= hi - rel_eps * rng))
    return float(sat / v.size)


def summarize(values, threshold: float) -> dict:
    """Convenience: bundle the three stats into one dict for the UI."""
    return {
        'snr_db': signal_snr_db(values, threshold),
        'sat_frac': saturated_fraction(values),
    }
