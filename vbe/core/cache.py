"""LRU frame cache and on-disk signal cache I/O.

The signal cache is keyed on a SHA-1 of (video_path, roi, channel, start_frame,
end_frame), truncated to 16 hex chars. The on-disk layout is:

    <video_dir>/.sig_cache/<key>.npz       # times, values
    <video_dir>/.sig_cache/<key>.json      # metadata

Metadata version 2 is the current schema.
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np


class FrameCache:
    """Tiny LRU cache for decoded video frames keyed by frame index."""

    def __init__(self, maxsize: int = 120):
        self._d: OrderedDict = OrderedDict()
        self._max = maxsize

    def get(self, key):
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        return None

    def put(self, key, value):
        if key in self._d:
            self._d.move_to_end(key)
        self._d[key] = value
        if len(self._d) > self._max:
            self._d.popitem(last=False)


def signal_cache_key(video_path: str, roi, channel: str,
                     start_frame: int, end_frame: int) -> str:
    payload = f'{video_path}|{tuple(roi)}|{channel}|{start_frame}|{end_frame}'
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _cache_dir(video_path: str) -> Path:
    d = Path(video_path).parent / '.sig_cache'
    d.mkdir(exist_ok=True)
    return d


def signal_cache_path(video_path: str, key: str) -> Path:
    return _cache_dir(video_path) / f'{key}.npz'


def signal_cache_meta_path(video_path: str, key: str) -> Path:
    return signal_cache_path(video_path, key).with_suffix('.json')


def load_signal_cache(video_path: str, key: str):
    p = signal_cache_path(video_path, key)
    if not p.exists():
        return None
    try:
        d = np.load(str(p))
        return d['times'], d['values']
    except Exception:
        p.unlink(missing_ok=True)
        return None


def save_signal_cache(video_path: str, key: str,
                      times: np.ndarray, values: np.ndarray,
                      metadata: dict | None = None) -> None:
    p = signal_cache_path(video_path, key)
    try:
        np.savez_compressed(
            str(p), times=times, values=values,
            metadata_json=json.dumps(metadata or {}))
        signal_cache_meta_path(video_path, key).write_text(
            json.dumps(metadata or {}, indent=2), encoding='utf-8')
    except Exception:
        pass


def load_cache_metadata(video_path: str, key: str):
    p = signal_cache_meta_path(video_path, key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None


def write_cache_metadata(video_path: str, key: str, metadata: dict) -> None:
    try:
        signal_cache_meta_path(video_path, key).write_text(
            json.dumps(metadata, indent=2), encoding='utf-8')
    except Exception:
        pass


def clear_signal_cache(video_path: str) -> int:
    cache_dir = Path(video_path).parent / '.sig_cache'
    if not cache_dir.exists():
        return 0
    n = 0
    for f in list(cache_dir.glob('*.npz')) + list(cache_dir.glob('*.json')):
        f.unlink(missing_ok=True)
        n += 1
    return n
