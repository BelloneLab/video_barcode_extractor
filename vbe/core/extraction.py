"""Video ROI extraction.

`extract_chunk` is the per-subprocess worker used by `ExtractionWorker`. It
must be a top-level callable in an importable module so Windows spawn-mode
`ProcessPoolExecutor` can pickle it.

A 60-frame lead-in compensates for OpenCV+FFmpeg snapping `CAP_PROP_POS_FRAMES`
to the nearest keyframe in long-GOP codecs. Pass B will replace this with
PyAV's frame-accurate seek API for stronger correctness guarantees.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

_CH_MAP = {'gray': None, 'r': 2, 'g': 1, 'b': 0}
_LUM_W = np.array([0.114, 0.587, 0.299], dtype=np.float32)  # B G R, BT.601
_LEAD_IN_FRAMES = 60


def extract_chunk(video_path: str, roi: tuple, start_frame: int,
                  end_frame: int, channel: str, decoder: str = 'opencv'):
    """Decode frames [start_frame, end_frame) and return (start_frame, values).

    The OpenCV decoder is fast and ubiquitous but its `CAP_PROP_POS_FRAMES`
    seek snaps to keyframes on long-GOP codecs. A 60-frame lead-in compensates.

    The PyAV decoder uses `av.open` and per-frame PTS, giving frame-accurate
    seeks on all containers. It is enabled when `decoder='pyav'` AND the `av`
    package is importable; otherwise the function silently falls back to
    OpenCV.
    """
    if decoder == 'pyav':
        try:
            return _extract_chunk_pyav(video_path, roi, start_frame,
                                       end_frame, channel)
        except Exception:
            pass  # fall through to OpenCV
    return _extract_chunk_opencv(video_path, roi, start_frame, end_frame,
                                 channel)


def _extract_chunk_opencv(video_path, roi, start_frame, end_frame, channel):
    ch_idx = _CH_MAP[channel]
    x, y, w, h = roi

    cap = cv2.VideoCapture(video_path)
    seek_to = max(0, start_frame - _LEAD_IN_FRAMES)
    cap.set(cv2.CAP_PROP_POS_FRAMES, seek_to)

    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    while pos < start_frame:
        if not cap.grab():
            break
        pos += 1

    n = end_frame - start_frame
    values = np.empty(n, dtype=np.float32)
    count = 0
    for i in range(n):
        ret, frame = cap.read()
        if not ret:
            break
        crop = frame[max(0, y): y + h, max(0, x): x + w]
        if crop.size == 0:
            values[i] = 0.0
        elif ch_idx is None:
            means = crop.mean(axis=(0, 1))
            values[i] = float(np.dot(means, _LUM_W))
        else:
            values[i] = float(crop[:, :, ch_idx].mean())
        count += 1
    cap.release()
    return start_frame, values[:count]


def _extract_chunk_pyav(video_path, roi, start_frame, end_frame, channel):
    """PyAV-based decoder. Uses the container's average frame rate to convert
    frame indices to PTS. For VFR videos the PTS itself is accurate; the
    frame-index mapping used by the rest of the app is still based on average
    fps for now. Future work: surface true PTS alongside values."""
    import av

    ch_idx = _CH_MAP[channel]
    x, y, w, h = roi

    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        avg_rate = float(stream.average_rate) if stream.average_rate else 30.0
        # Seek by absolute time in the stream's time_base. Seeking to a slightly
        # earlier point and decoding forward gives frame-accurate landing on
        # the target frame across containers.
        seek_to_frame = max(0, start_frame - _LEAD_IN_FRAMES)
        seek_pts = int(seek_to_frame / avg_rate / stream.time_base)
        container.seek(seek_pts, stream=stream, any_frame=False, backward=True)

        n = end_frame - start_frame
        values = np.empty(n, dtype=np.float32)
        count = 0
        for frame in container.decode(stream):
            frame_idx = int(round(float(frame.pts * stream.time_base) * avg_rate))
            if frame_idx < start_frame:
                continue
            if frame_idx >= end_frame or count >= n:
                break
            arr = frame.to_ndarray(format='bgr24')
            crop = arr[max(0, y): y + h, max(0, x): x + w]
            if crop.size == 0:
                values[count] = 0.0
            elif ch_idx is None:
                means = crop.mean(axis=(0, 1))
                values[count] = float(np.dot(means, _LUM_W))
            else:
                values[count] = float(crop[:, :, ch_idx].mean())
            count += 1
        return start_frame, values[:count]
    finally:
        container.close()


class ExtractionWorker(QThread):
    """Background extraction over a frame range using a process pool."""

    progress = pyqtSignal(int)
    finished = pyqtSignal(object, object)
    error = pyqtSignal(str)

    def __init__(self, video_path, roi, start_frame, end_frame,
                 channel, fps, n_workers=1, decoder='opencv'):
        super().__init__()
        self.video_path = video_path
        self.roi = roi
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.channel = channel
        self.fps = fps
        self.n_workers = max(1, n_workers)
        self.decoder = decoder
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            total = self.end_frame - self.start_frame
            n_workers = min(self.n_workers, max(1, total))
            chunk_frames = max(1, min(300, total // max(n_workers * 4, 1)))

            chunks = []
            f = self.start_frame
            while f < self.end_frame:
                e = min(f + chunk_frames, self.end_frame)
                chunks.append((f, e))
                f = e

            results = {}
            done_count = 0
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                future_map = {
                    pool.submit(extract_chunk, self.video_path, self.roi,
                                s, e, self.channel, self.decoder): s
                    for s, e in chunks
                }
                for fut in as_completed(future_map):
                    if self._abort:
                        for pending in future_map:
                            pending.cancel()
                        return
                    chunk_start, vals = fut.result()
                    results[chunk_start] = vals
                    done_count += len(vals)
                    self.progress.emit(self.start_frame + done_count)

            ordered = [results[s] for s, _ in chunks if s in results]
            if not ordered:
                self.error.emit('No frames extracted.')
                return
            arr = np.concatenate(ordered).astype(np.float32)
            times = (np.arange(len(arr)) + self.start_frame) / self.fps
            self.finished.emit(times, arr)
        except Exception as exc:
            self.error.emit(str(exc))
