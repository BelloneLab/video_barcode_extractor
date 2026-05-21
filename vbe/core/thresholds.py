"""Thresholding methods for extracted ROI signals.

Otsu is the default and works for roughly balanced classes (on:off ~ 1:1).
For sparse pulses (e.g. 5% duty-cycle barcode flashes) Otsu chooses a
threshold biased toward the dominant class. We expose four alternatives:

- Triangle (Zack 1977): geometric construction robust to skewed histograms.
- Kapur entropy: maximum-entropy threshold, principled for sparse signals.
- MAD (median absolute deviation): robust statistical fence.
- Percentile: tunable, simple, sometimes the right thing.

The Auto method inspects class balance and picks Otsu when the signal looks
balanced, Triangle otherwise.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def _as_finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return x[np.isfinite(x)]


def threshold_otsu(x) -> float:
    """Otsu's between-class-variance maximization."""
    from skimage.filters import threshold_otsu as _otsu
    arr = _as_finite(x)
    if arr.size < 2:
        return 0.0
    return float(_otsu(arr))


def threshold_triangle(x) -> float:
    """Triangle method (Zack et al. 1977). Good for skewed histograms."""
    from skimage.filters import threshold_triangle as _tri
    arr = _as_finite(x)
    if arr.size < 2:
        return 0.0
    return float(_tri(arr))


def threshold_kapur(x, n_bins: int = 256) -> float:
    """Maximum-entropy threshold (Kapur, Sahoo, Wong 1985)."""
    arr = _as_finite(x)
    if arr.size < 2:
        return 0.0
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return lo
    hist, edges = np.histogram(arr, bins=n_bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    eps = 1e-12

    H0 = np.zeros(n_bins, dtype=np.float64)
    H1 = np.zeros(n_bins, dtype=np.float64)
    for t in range(n_bins):
        p_lo, P_lo = p[:t + 1], p[:t + 1].sum()
        if P_lo > eps:
            r = p_lo[p_lo > 0] / P_lo
            H0[t] = -np.sum(r * np.log(r))
        p_hi, P_hi = p[t + 1:], p[t + 1:].sum()
        if P_hi > eps:
            r = p_hi[p_hi > 0] / P_hi
            H1[t] = -np.sum(r * np.log(r))
    phi = H0 + H1
    return float(centers[int(np.argmax(phi))])


def threshold_mad(x, k: float = 3.0) -> float:
    """Robust off-state baseline + k sigma.

    MAD is computed on samples at or below the median (the assumed off-state),
    then scaled by 1.4826 to give a std-equivalent for Gaussian noise. For
    balanced bimodal data this still yields a useful threshold because the
    off-state baseline noise is bounded; for sparse pulses it is exactly the
    right model.
    """
    arr = _as_finite(x)
    if arr.size < 2:
        return 0.0
    med = float(np.median(arr))
    off = arr[arr <= med]
    if off.size < 2:
        return med
    med_off = float(np.median(off))
    mad_off = float(np.median(np.abs(off - med_off)))
    return med_off + k * mad_off * 1.4826


def threshold_percentile(x, p: float = 95.0) -> float:
    """The p-th percentile of the signal. Cheap, tunable, often surprisingly good."""
    arr = _as_finite(x)
    if arr.size < 2:
        return 0.0
    return float(np.percentile(arr, p))


def auto_threshold(x) -> Tuple[float, str]:
    """Pick a method based on class balance.

    Class balance is approximated by the fraction of samples above the
    mid-range value (min+max)/2. Median is a poor proxy here because it is
    50% by definition; mid-range tracks the duty cycle of a roughly bimodal
    signal. Below 15% or above 85% on-fraction we switch from Otsu to
    Triangle, which is more robust to skewed histograms.
    """
    arr = _as_finite(x)
    if arr.size < 2:
        return 0.0, 'Otsu'
    mid = 0.5 * (float(arr.min()) + float(arr.max()))
    frac_high = float(np.mean(arr > mid))
    if frac_high < 0.15 or frac_high > 0.85:
        return threshold_triangle(arr), 'Triangle'
    return threshold_otsu(arr), 'Otsu'


METHODS = {
    'Otsu':       threshold_otsu,
    'Triangle':   threshold_triangle,
    'Kapur':      threshold_kapur,
    'MAD':        threshold_mad,
    'Percentile': threshold_percentile,
}
