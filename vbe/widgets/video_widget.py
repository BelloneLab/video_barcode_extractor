"""Video display widget with ROI drawing, scroll-wheel zoom, and middle/Ctrl pan."""
from __future__ import annotations

import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QSizePolicy, QWidget

from vbe.theme import BG2, BLUE, FIT_MARGIN


class VideoWidget(QWidget):
    roiChanged = pyqtSignal(tuple)   # (x, y, w, h) in video-pixel coords
    zoomChanged = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 340)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)

        self._pixmap = None
        self._vid_w = 1
        self._vid_h = 1
        self._scale = 1.0
        self._offset = QPointF(0.0, 0.0)

        self._roi = None
        self._draw = False
        self._dp0 = QPointF()
        self._dp1 = QPointF()

        self._panning = False
        self._pan_org = QPointF()

    def _w2v(self, p: QPointF) -> QPointF:
        return QPointF((p.x() - self._offset.x()) / self._scale,
                       (p.y() - self._offset.y()) / self._scale)

    def _v2w(self, p: QPointF) -> QPointF:
        return QPointF(p.x() * self._scale + self._offset.x(),
                       p.y() * self._scale + self._offset.y())

    def _clamp(self, p: QPointF) -> QPointF:
        return QPointF(max(0.0, min(float(self._vid_w - 1), p.x())),
                       max(0.0, min(float(self._vid_h - 1), p.y())))

    def set_frame(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        self._vid_w, self._vid_h = w, h
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(img)
        self.update()

    def fit_to_widget(self):
        if not self._pixmap:
            return
        s = min(self.width() / self._vid_w, self.height() / self._vid_h) * FIT_MARGIN
        self._scale = s
        self._offset = QPointF((self.width()  - self._vid_w * s) / 2.0,
                               (self.height() - self._vid_h * s) / 2.0)
        self.update()
        self.zoomChanged.emit(self._scale)

    def zoom_by(self, factor: float, center: QPointF = None):
        if center is None:
            center = QPointF(self.width() / 2.0, self.height() / 2.0)
        v = self._w2v(center)
        self._scale = max(0.05, min(30.0, self._scale * factor))
        self._offset = QPointF(center.x() - v.x() * self._scale,
                               center.y() - v.y() * self._scale)
        self.update()
        self.zoomChanged.emit(self._scale)

    def set_zoom_100(self):
        self._scale = 1.0
        self._offset = QPointF((self.width()  - self._vid_w) / 2.0,
                               (self.height() - self._vid_h) / 2.0)
        self.update()
        self.zoomChanged.emit(self._scale)

    def set_roi(self, roi):
        self._roi = roi
        self.update()

    def clear_roi(self):
        self._roi = None
        self.update()

    def get_roi(self):
        return self._roi

    def wheelEvent(self, e):
        f = 1.15 if e.angleDelta().y() > 0 else 1.0 / 1.15
        self.zoom_by(f, QPointF(e.pos()))

    def mousePressEvent(self, e):
        is_pan = (e.button() == Qt.MiddleButton or
                  (e.button() == Qt.LeftButton and
                   e.modifiers() & Qt.ControlModifier))
        if is_pan:
            self._panning = True
            self._pan_org = QPointF(e.pos())
            self.setCursor(Qt.ClosedHandCursor)
        elif e.button() == Qt.LeftButton:
            self._draw = True
            self._dp0 = self._clamp(self._w2v(QPointF(e.pos())))
            self._dp1 = self._dp0

    def mouseMoveEvent(self, e):
        if self._panning:
            d = QPointF(e.pos()) - self._pan_org
            self._offset += d
            self._pan_org = QPointF(e.pos())
            self.update()
        elif self._draw:
            self._dp1 = self._clamp(self._w2v(QPointF(e.pos())))
            self.update()

    def mouseReleaseEvent(self, e):
        released_pan = (e.button() == Qt.MiddleButton or
                        (e.button() == Qt.LeftButton and self._panning))
        if self._panning and released_pan:
            self._panning = False
            self.setCursor(Qt.CrossCursor)
        elif self._draw and e.button() == Qt.LeftButton:
            self._draw = False
            ix = int(min(self._dp0.x(), self._dp1.x()))
            iy = int(min(self._dp0.y(), self._dp1.y()))
            iw = int(abs(self._dp1.x() - self._dp0.x()))
            ih = int(abs(self._dp1.y() - self._dp0.y()))
            if iw > 2 and ih > 2:
                self._roi = (ix, iy, iw, ih)
                self.roiChanged.emit(self._roi)
            self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(BG2))

        if self._pixmap:
            p.save()
            p.translate(self._offset)
            p.scale(self._scale, self._scale)
            p.drawPixmap(0, 0, self._pixmap)
            p.restore()

        if self._roi:
            rx, ry, rw, rh = self._roi
            tl = self._v2w(QPointF(rx, ry))
            br = self._v2w(QPointF(rx + rw, ry + rh))
            rect = QRectF(tl, br)
            p.setPen(QPen(QColor('#ff4081'), 2, Qt.SolidLine))
            p.drawRect(rect)
            sz = 5.0
            p.setPen(QPen(QColor('#ff80ab'), 2))
            for cx, cy in [(tl.x(), tl.y()), (br.x(), tl.y()),
                           (tl.x(), br.y()), (br.x(), br.y())]:
                p.drawRect(QRectF(cx - sz / 2, cy - sz / 2, sz, sz))
            p.setPen(QPen(QColor('#ff80ab')))
            p.setFont(QFont('Consolas', 9))
            p.drawText(QPointF(tl.x() + 3, tl.y() - 5),
                       f'ROI  {rw}x{rh}  px')

        if self._draw:
            tl = self._v2w(self._dp0)
            br = self._v2w(self._dp1)
            p.setPen(QPen(QColor(BLUE), 1.5, Qt.DashLine))
            p.drawRect(QRectF(tl, br))

        p.end()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._pixmap:
            self.fit_to_widget()
