"""Unit tests for the LRU frame cache and signal-cache key/path helpers."""
from __future__ import annotations

import numpy as np
import pytest

from vbe.core.cache import (
    FrameCache, load_signal_cache, save_signal_cache, signal_cache_key,
    signal_cache_path,
)


# ---------------------------------------------------------------------------
# signal_cache_key
# ---------------------------------------------------------------------------
def test_cache_key_stable_under_repeated_calls():
    args = ('/some/path.mp4', (10, 20, 100, 50), 'gray', 0, 1000)
    a = signal_cache_key(*args)
    b = signal_cache_key(*args)
    assert a == b
    assert len(a) == 16


def test_cache_key_changes_with_roi():
    a = signal_cache_key('/x.mp4', (0, 0, 10, 10), 'gray', 0, 100)
    b = signal_cache_key('/x.mp4', (0, 0, 11, 10), 'gray', 0, 100)
    assert a != b


def test_cache_key_changes_with_channel():
    a = signal_cache_key('/x.mp4', (0, 0, 10, 10), 'gray', 0, 100)
    b = signal_cache_key('/x.mp4', (0, 0, 10, 10), 'r', 0, 100)
    assert a != b


def test_cache_key_changes_with_range():
    a = signal_cache_key('/x.mp4', (0, 0, 10, 10), 'gray', 0, 100)
    b = signal_cache_key('/x.mp4', (0, 0, 10, 10), 'gray', 0, 101)
    assert a != b


# ---------------------------------------------------------------------------
# FrameCache LRU
# ---------------------------------------------------------------------------
def test_frame_cache_put_get():
    c = FrameCache(maxsize=3)
    c.put(0, 'a')
    c.put(1, 'b')
    assert c.get(0) == 'a'
    assert c.get(1) == 'b'
    assert c.get(99) is None


def test_frame_cache_evicts_least_recently_used():
    c = FrameCache(maxsize=2)
    c.put(0, 'a')
    c.put(1, 'b')
    _ = c.get(0)        # 'a' becomes most recent
    c.put(2, 'c')       # should evict 'b', not 'a'
    assert c.get(0) == 'a'
    assert c.get(2) == 'c'
    assert c.get(1) is None


def test_frame_cache_put_existing_key_refreshes_order():
    c = FrameCache(maxsize=2)
    c.put(0, 'a')
    c.put(1, 'b')
    c.put(0, 'A')   # update; 'A' is now most recent
    c.put(2, 'c')   # should evict 'b'
    assert c.get(0) == 'A'
    assert c.get(1) is None
    assert c.get(2) == 'c'


# ---------------------------------------------------------------------------
# save_signal_cache / load_signal_cache round-trip on disk
# ---------------------------------------------------------------------------
def test_save_then_load_signal_cache_roundtrip(tmp_path):
    # Set up a fake "video" file so the cache helper can derive its cache dir.
    fake_video = tmp_path / 'clip.mp4'
    fake_video.write_bytes(b'\x00')
    key = signal_cache_key(str(fake_video), (0, 0, 10, 10), 'gray', 0, 500)
    times = np.linspace(0, 5, 100, dtype=np.float64)
    values = np.sin(times * 2 * np.pi).astype(np.float32)

    save_signal_cache(str(fake_video), key, times, values, metadata={'foo': 1})
    cached = load_signal_cache(str(fake_video), key)
    assert cached is not None
    t_out, v_out = cached
    assert np.allclose(t_out, times)
    assert np.allclose(v_out, values)

    # The cache directory should exist next to the fake video
    assert signal_cache_path(str(fake_video), key).exists()
