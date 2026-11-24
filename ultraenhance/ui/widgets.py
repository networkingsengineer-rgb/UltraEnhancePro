"""Reusable widgets: drop zone and before/after comparison view."""
from __future__ import annotations

from typing import Callable, List, Optional

from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget


def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    img = img.convert("RGBA")
    data = img.tobytes("raw", "RGBA")
    qimg = QImage(data, img.width, img.height, img.width * 4, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


class DropArea(QFrame):
    """Dashed drop zone; emits the list of dropped local paths."""
    dropped = Signal(list)
    clicked = Signal()

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.setObjectName("drop")
        self.setAcceptDrops(True)
        self.setMinimumHeight(120)
        lay = QVBoxLayout(self)
        self.label = QLabel(text)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(True)
        lay.addWidget(self.label)
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, e):
        self.clicked.emit()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.dropped.emit(paths)
            e.acceptProposedAction()


class CompareView(QWidget):
    """Before/after split slider with mouse-wheel zoom and drag-to-pan.

    * Move the mouse with the left button near the split line to move it.
    * Drag elsewhere to pan. Wheel to zoom (cursor-centred). Double-click: fit.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.before: Optional[QPixmap] = None
        self.after: Optional[QPixmap] = None
        self.split = 0.5
        self.zoom = 1.0
        self.offset = QPointF(0, 0)
        self._drag_mode = None
        self._last = QPointF()
        self.setMinimumSize(360, 260)
        self.setMouseTracking(True)
        self.placeholder = "Select a photo and press Preview"

    def set_images(self, before: Optional[QPixmap], after: Optional[QPixmap]):
        self.before, self.after = before, after
        self.fit()

    def fit(self):
        pm = self.after or self.before
        if pm and pm.width() and pm.height():
            self.zoom = min(self.width() / pm.width(), self.height() / pm.height())
            self.offset = QPointF((self.width() - pm.width() * self.zoom) / 2,
                                  (self.height() - pm.height() * self.zoom) / 2)
        self.update()

    def resizeEvent(self, e):
        self.fit()
        super().resizeEvent(e)

    def _img_rect(self) -> QRectF:
        pm = self.after or self.before
        if not pm:
            return QRectF()
        return QRectF(self.offset.x(), self.offset.y(), pm.width() * self.zoom, pm.height() * self.zoom)

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0f1115"))
        if not (self.before or self.after):
            p.setPen(QColor("#8a90a0"))
            p.drawText(self.rect(), Qt.AlignCenter, self.placeholder)
            return
        p.setRenderHint(QPainter.SmoothPixmapTransform, self.zoom < 2)
        r = self._img_rect()
        split_x = r.left() + r.width() * self.split
        if self.before:
            p.save()
            p.setClipRect(QRectF(r.left(), r.top(), split_x - r.left(), r.height()))
            p.drawPixmap(r, self.before, QRectF(self.before.rect()))
            p.restore()
        if self.after:
            p.save()
            p.setClipRect(QRectF(split_x, r.top(), r.right() - split_x, r.height()))
            p.drawPixmap(r, self.after, QRectF(self.after.rect()))
            p.restore()
        p.setPen(QPen(QColor("white"), 2))
        p.drawLine(QPointF(split_x, max(0, r.top())), QPointF(split_x, min(self.height(), r.bottom())))
        p.setBrush(QColor("#6c5ce7"))
        p.drawEllipse(QPointF(split_x, self.height() / 2), 9, 9)
        p.setPen(QColor("white"))
        p.drawText(12, 22, "Before")
        p.drawText(self.width() - 52, 22, "After")
        p.drawText(12, self.height() - 10, f"Zoom {self.zoom * 100:.0f}%  (wheel to zoom, double-click to fit)")

    def mousePressEvent(self, e):
        r = self._img_rect()
        split_x = r.left() + r.width() * self.split
        pos = e.position()
        self._drag_mode = "split" if abs(pos.x() - split_x) < 14 else "pan"
        self._last = pos

    def mouseMoveEvent(self, e):
        pos = e.position()
        r = self._img_rect()
        if self._drag_mode == "split" and r.width() > 0:
            self.split = max(0.0, min(1.0, (pos.x() - r.left()) / r.width()))
            self.update()
        elif self._drag_mode == "pan":
            self.offset += pos - self._last
            self._last = pos
            self.update()
        else:
            split_x = r.left() + r.width() * self.split
            self.setCursor(Qt.SplitHCursor if abs(pos.x() - split_x) < 14 else Qt.OpenHandCursor)

    def mouseReleaseEvent(self, e):
        self._drag_mode = None

    def mouseDoubleClickEvent(self, e):
        self.fit()

    def wheelEvent(self, e):
        factor = 1.25 if e.angleDelta().y() > 0 else 0.8
        new_zoom = max(0.05, min(16.0, self.zoom * factor))
        pos = e.position()
        self.offset = pos - (pos - self.offset) * (new_zoom / self.zoom)
        self.zoom = new_zoom
        self.update()
