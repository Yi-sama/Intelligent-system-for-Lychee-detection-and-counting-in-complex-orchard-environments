from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

from PyQt5.QtCore import QObject, QPoint, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QScrollArea,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.export import (
    export_annotated_image,
    export_annotated_images,
    export_annotated_video,
    export_detections_csv,
    export_detections_csv_batch,
    export_video_detections_csv,
)
from app.infer import CameraFrameResult, DetectionResult, DetectionService, VideoDetectionResult, VideoFrameResult

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - depends on local env
    cv2 = None


TEXT = {
    "image_mode": "\u56fe\u7247\u6a21\u5f0f",
    "video_mode": "\u89c6\u9891\u6a21\u5f0f",
    "camera_mode": "\u6444\u50cf\u5934\u6a21\u5f0f",
    "dialog_title": "\u56fe\u7247\u9884\u89c8",
    "dialog_empty": "\u6682\u65e0\u53ef\u9884\u89c8\u7684\u56fe\u7247",
    "image_config": "\u56fe\u7247\u914d\u7f6e",
    "image_source": "\u56fe\u7247\u6765\u6e90",
    "single_image": "\u5355\u5f20\u56fe\u7247",
    "batch_folder": "\u591a\u56fe\u6279\u91cf",
    "confidence": "\u7f6e\u4fe1\u5ea6",
    "iou": "\u4ea4\u5e76\u6bd4",
    "label_visibility": "\u6807\u7b7e\u663e\u793a",
    "show_labels": "\u663e\u793a\u6807\u7b7e",
    "export_format": "\u5bfc\u51fa\u683c\u5f0f",
    "export_both": "CSV + \u7ed3\u679c\u56fe",
    "export_csv_only": "\u4ec5 CSV",
    "export_image_only": "\u4ec5\u7ed3\u679c\u56fe",
    "export_video_only": "\u4ec5\u7ed3\u679c\u89c6\u9891",
    "video_reserved": "\u89c6\u9891\u6a21\u5f0f\u5165\u53e3\u5df2\u4fdd\u7559",
    "video_config": "\u89c6\u9891\u914d\u7f6e",
    "video_source": "\u89c6\u9891\u6587\u4ef6",
    "video_loaded": "\u89c6\u9891\u5df2\u52a0\u8f7d\uff0c\u70b9\u51fb\"\u5f00\u59cb\u68c0\u6d4b\"\u6267\u884c\u9010\u5e27\u63a8\u7406\u3002",
    "video_ready": "\u89c6\u9891\u5df2\u5c31\u7eea",
    "video_processing": "\u89c6\u9891\u5904\u7406\u4e2d",
    "video_done": "\u89c6\u9891\u68c0\u6d4b\u5b8c\u6210",
    "video_pick_title": "\u9009\u62e9\u5f85\u68c0\u6d4b\u89c6\u9891",
    "need_video_title": "\u672a\u9009\u62e9\u89c6\u9891",
    "need_video_body": "\u8bf7\u5148\u6253\u5f00\u4e00\u4e2a mp4/avi/mov/mkv \u89c6\u9891\u6587\u4ef6\u3002",
    "video_status_idle": "\u652f\u6301\u672c\u5730\u89c6\u9891\u9010\u5e27\u68c0\u6d4b\uff0c\u7ed3\u679c\u4f1a\u5728\u53f3\u4fa7\u9884\u89c8\u533a\u548c\u4e0b\u65b9\u7ed3\u679c\u533a\u66f4\u65b0\u3002",
    "video_status_running": "\u89c6\u9891\u6b63\u5728\u5904\u7406\uff0cGUI \u4fdd\u6301\u54cd\u5e94\uff0c\u53ef\u5728\u9884\u89c8\u533a\u67e5\u770b\u5f53\u524d\u5df2\u5904\u7406\u5e27\u3002",
    "video_status_done": "\u89c6\u9891\u68c0\u6d4b\u5b8c\u6210\uff0c\u53ef\u5bfc\u51fa CSV \u6216\u6807\u6ce8\u89c6\u9891\u3002",
    "video_overview_ready": "\u672c\u5730\u89c6\u9891\u68c0\u6d4b\u5c1a\u672a\u5f00\u59cb\u3002",
    "video_overview_progress": "\u5df2\u5904\u7406\u5e27\uff1a{processed}/{total}",
    "video_overview_total": "\u89c6\u9891\u603b\u68c0\u6d4b\u6570\uff1a{count}",
    "video_overview_frames": "\u89c6\u9891\u5e27\u6570\uff1a{processed}/{total}",
    "video_detail_line": "\u7b2c {frame_index} \u5e27 | {label} | \u7f6e\u4fe1\u5ea6 {confidence:.2f} | \u6846 {bbox}",
    "video_frame_preview_unavailable": "\u5df2\u52a0\u8f7d\u89c6\u9891\uff0c\u4f46\u5f53\u524d\u73af\u5883\u65e0\u6cd5\u63d0\u53d6\u9884\u89c8\u5e27\u3002",
    "video_running_block_title": "\u89c6\u9891\u68c0\u6d4b\u8fdb\u884c\u4e2d",
    "video_running_block_body": "\u8bf7\u7b49\u5f85\u5f53\u524d\u89c6\u9891\u68c0\u6d4b\u5b8c\u6210\u540e\u518d\u5207\u6362\u6a21\u5f0f\u6216\u91cd\u65b0\u6253\u5f00\u6587\u4ef6\u3002",
    "export_video_body": "\u6807\u6ce8\u89c6\u9891\u5df2\u4fdd\u5b58\u5230:\n{path}",
    "camera_reserved": "\u6444\u50cf\u5934\u6a21\u5f0f\u5165\u53e3\u5df2\u4fdd\u7559",
    "camera_device": "\u6444\u50cf\u5934\u8bbe\u5907",
    "camera_device_default": "\u9ed8\u8ba4\u6444\u50cf\u5934 (0)",
    "camera_save_frames": "\u4fdd\u5b58\u622a\u56fe",
    "camera_status_ready": "\u5c31\u7eea\u540e\u53ef\u6253\u5f00\u6444\u50cf\u5934\u5e76\u542f\u52a8\u5b9e\u65f6\u68c0\u6d4b\u3002",
    "camera_status_opened": "\u6444\u50cf\u5934\u5df2\u6253\u5f00\uff0c\u53ef\u5f00\u59cb\u5b9e\u65f6\u68c0\u6d4b\u3002",
    "camera_status_running": "\u6444\u50cf\u5934\u68c0\u6d4b\u8fdb\u884c\u4e2d\uff0c\u6b63\u5728\u51c6\u5907\u52a8\u6001\u8ba1\u6570\u3002",
    "camera_status_stopped": "\u6444\u50cf\u5934\u68c0\u6d4b\u5df2\u505c\u6b62\uff0c\u53ef\u91cd\u65b0\u5f00\u59cb\u3002",
    "camera_status_preview_error": "\u6444\u50cf\u5934\u753b\u9762\u8bfb\u53d6\u5931\u8d25\uff0c\u8bf7\u68c0\u67e5\u8bbe\u5907\u662f\u5426\u88ab\u5360\u7528\u3002",
    "camera_status_detect_requires_preview": "\u8bf7\u5148\u6253\u5f00\u6444\u50cf\u5934\u9884\u89c8\uff0c\u518d\u542f\u52a8\u5b9e\u65f6\u68c0\u6d4b\u3002",
    "camera_overview_ready": "\u6444\u50cf\u5934\u5b9e\u65f6\u68c0\u6d4b\u5c1a\u672a\u5f00\u59cb\u3002",
    "camera_overview_running": "\u5b9e\u65f6\u68c0\u6d4b\u5df2\u542f\u52a8\uff0c\u5c06\u5728\u6b64\u5904\u663e\u793a\u52a8\u6001\u8ba1\u6570\u3002",
    "camera_overview_stopped": "\u5b9e\u65f6\u68c0\u6d4b\u5df2\u505c\u6b62\uff0c\u53ef\u91cd\u65b0\u542f\u52a8\u52a8\u6001\u8ba1\u6570\u3002",
    "camera_preview_waiting": "\u6b63\u5728\u7b49\u5f85\u6444\u50cf\u5934\u753b\u9762...",
    "camera_overview_total": "\u5f53\u524d\u5e27\u68c0\u6d4b\u6570\uff1a{count}",
    "camera_overview_saved": "\u5df2\u4fdd\u5b58\u81f3\uff1a{path}",
    "camera_frame_name": "camera_frame.jpg",
    "model": "\u6a21\u578b",
    "current_file": "\u5f53\u524d\u6587\u4ef6",
    "state": "\u72b6\u6001",
    "elapsed": "\u8017\u65f6",
    "ready": "\u56fe\u7247\u5df2\u5c31\u7eea",
    "running": "\u68c0\u6d4b\u4e2d",
    "done": "\u68c0\u6d4b\u5b8c\u6210",
    "failed": "\u68c0\u6d4b\u5931\u8d25",
    "open": "\u6253\u5f00",
    "open_camera": "\u6253\u5f00\u6444\u50cf\u5934",
    "close_camera": "\u5173\u95ed\u6444\u50cf\u5934",
    "detect": "\u5f00\u59cb\u68c0\u6d4b",
    "stop_detect": "\u505c\u6b62\u68c0\u6d4b",
    "preview": "\u9884\u89c8",
    "preview_previous": "\u25c0",
    "preview_next": "\u25b6",
    "toggle": "\u539f\u56fe/\u7ed3\u679c",
    "video_original": "\u539f\u56fe",
    "video_result": "\u7ed3\u679c\u56fe",
    "video_restart": "\u91cd\u65b0\u5f00\u59cb",
    "export": "\u5bfc\u51fa",
    "close_current": "\u2715",
    "close_all": "\u5168\u90e8\u5173\u95ed",
    "preview_hint": "\u70b9\u51fb\u56fe\u7247\u5207\u6362\u539f\u56fe/\u7ed3\u679c\uff0c\u70b9\u9884\u89c8\u67e5\u770b\u5927\u56fe",
    "video_preview_hint": "\u70b9\u51fb\u89c6\u9891\u6682\u505c/\u7ee7\u7eed\uff0c\u6682\u505c\u540e\u53ef\u5207\u6362\u539f\u56fe/\u7ed3\u679c\u56fe\uff0c\u70b9\u9884\u89c8\u67e5\u770b\u5927\u56fe",
    "preview_empty": "\u9009\u62e9\u56fe\u7247\u540e\u5f00\u59cb\u68c0\u6d4b",
    "preview_closed": "\u5f53\u524d\u663e\u793a\u5df2\u5173\u95ed",
    "summary_title": "\u68c0\u6d4b\u6982\u89c8",
    "detail_title": "\u68c0\u6d4b\u660e\u7ec6",
    "no_result": "\u5f53\u524d\u6ca1\u6709\u68c0\u6d4b\u7ed3\u679c",
    "result_closed": "\u5f53\u524d\u663e\u793a\u7ed3\u679c\u5df2\u5173\u95ed",
    "pick_image_title": "\u9009\u62e9\u5f85\u68c0\u6d4b\u56fe\u7247",
    "pick_batch_title": "\u9009\u62e9\u591a\u5f20\u5f85\u68c0\u6d4b\u56fe\u7247",
    "need_image_title": "\u672a\u9009\u62e9\u56fe\u7247",
    "need_image_body": "\u8bf7\u5148\u6253\u5f00\u4e00\u5f20\u56fe\u7247\u3002",
    "need_batch_title": "\u672a\u9009\u62e9\u6279\u91cf\u56fe\u7247",
    "need_batch_body": "\u8bf7\u5148\u9009\u62e9\u4e00\u5f20\u6216\u591a\u5f20\u56fe\u7247\u3002",
    "batch_empty_title": "\u672a\u9009\u4e2d\u53ef\u7528\u56fe\u7247",
    "batch_empty_body": "\u8bf7\u81f3\u5c11\u9009\u62e9\u4e00\u5f20 png/jpg/jpeg/bmp \u56fe\u7247\u3002",
    "mode_limited_title": "\u6a21\u5f0f\u9650\u5236",
    "image_only_flow": "\u5f53\u524d\u4ec5\u652f\u6301\u56fe\u7247\u68c0\u6d4b\u5de5\u4f5c\u6d41\u3002",
    "other_modes_unavailable": "\u89c6\u9891\u548c\u6444\u50cf\u5934\u6a21\u5f0f\u6682\u672a\u5b9e\u73b0\u68c0\u6d4b\u6d41\u7a0b\u3002",
    "batch_placeholder": "\u6279\u91cf\u56fe\u7247\u5165\u53e3\u5df2\u4fdd\u7559\uff0c\u5f53\u524d\u4f18\u5148\u6253\u901a\u5355\u5f20\u56fe\u7247\u6d41\u7a0b\u3002",
    "image_loaded": "\u56fe\u7247\u5df2\u52a0\u8f7d\uff0c\u70b9\u51fb\"\u5f00\u59cb\u68c0\u6d4b\"\u5373\u53ef\u751f\u6210\u7ed3\u679c\u3002",
    "batch_loaded": "\u5df2\u52a0\u8f7d {count} \u5f20\u56fe\u7247\uff0c\u70b9\u51fb\"\u5f00\u59cb\u68c0\u6d4b\"\u6267\u884c\u6279\u91cf\u63a8\u7406\u3002",
    "export_missing_title": "\u6ca1\u6709\u53ef\u5bfc\u51fa\u7684\u7ed3\u679c",
    "export_missing_body": "\u8bf7\u5148\u5b8c\u6210\u4e00\u6b21\u68c0\u6d4b\u3002",
    "export_done": "\u5bfc\u51fa\u5b8c\u6210",
    "export_csv_body": "CSV \u5df2\u4fdd\u5b58\u5230:\n{path}",
    "export_image_body": "\u7ed3\u679c\u56fe\u5df2\u4fdd\u5b58\u5230:\n{path}",
    "export_batch_image_body": "\u6279\u91cf\u7ed3\u679c\u56fe\u5df2\u4fdd\u5b58\u5230:\n{path}",
    "preview_missing_title": "\u6682\u65e0\u7ed3\u679c",
    "preview_missing_body": "\u8bf7\u5148\u5b8c\u6210\u4e00\u6b21\u68c0\u6d4b\u3002",
    "camera_preview_body": "\u5f53\u524d\u6444\u50cf\u5934\u753b\u9762\u5df2\u5728\u9884\u89c8\u533a\u663e\u793a\uff0c\u70b9\u51fb\u53ef\u67e5\u770b\u5927\u56fe\u3002",
    "camera_unavailable_title": "\u65e0\u6cd5\u6253\u5f00\u6444\u50cf\u5934",
    "camera_unavailable_body": "\u5f53\u524d\u73af\u5883\u672a\u5b89\u88c5 OpenCV \u6216\u65e0\u6cd5\u8bbf\u95ee\u672c\u5730\u6444\u50cf\u5934\u3002",
    "camera_busy_body": "\u6444\u50cf\u5934\u6253\u5f00\u5931\u8d25\uff0c\u8bf7\u68c0\u67e5\u8bbe\u5907\u662f\u5426\u88ab\u5176\u4ed6\u7a0b\u5e8f\u5360\u7528\u3002",
    "display_closed_title": "\u5f53\u524d\u663e\u793a\u5df2\u5173\u95ed",
    "display_closed_body": "\u5f53\u524d\u9884\u89c8\u4e0e\u7ed3\u679c\u5df2\u88ab\u9690\u85cf\uff0c\u5207\u6362\u6a21\u5f0f\u6216\u56fe\u7247\u6765\u6e90\u540e\u53ef\u6062\u590d\u3002",
    "overview_total": "\u603b\u68c0\u6d4b\u6570\uff1a{count}",
    "overview_batch": "\u6279\u91cf\u56fe\u7247\uff1a{count}",
    "overview_none": "\u672a\u68c0\u6d4b\u5230\u76ee\u6807",
    "detail_line": "{label} | \u7f6e\u4fe1\u5ea6 {confidence:.2f} | \u6846 {bbox}",
    "detail_batch_line": "{file_name} | {label} | \u7f6e\u4fe1\u5ea6 {confidence:.2f} | \u6846 {bbox}",
}


def _preview_stage_min_length() -> int:
    screen = QApplication.primaryScreen()
    if screen is None:
        return 1200
    geometry = screen.availableGeometry()
    return max(900, int(min(geometry.width(), geometry.height()) * 0.75))


def _to_qpixmap(image_data: object) -> QPixmap:
    if image_data is None:
        return QPixmap()
    if isinstance(image_data, QPixmap):
        return image_data
    if isinstance(image_data, bytes):
        return QPixmap.fromImage(QImage.fromData(image_data))
    if hasattr(image_data, "shape"):
        height, width, channels = image_data.shape
        bytes_per_line = channels * width
        qimage = QImage(image_data.tobytes(), width, height, bytes_per_line, QImage.Format_BGR888)
        return QPixmap.fromImage(qimage.copy())
    raise TypeError(f"Unsupported image data type: {type(image_data)!r}")


class InteractiveImageLabel(QLabel):
    def __init__(self, text: str = "", on_zoom=None, on_click=None, parent: QWidget | None = None):
        super().__init__(text, parent)
        self._on_zoom = on_zoom
        self._on_click = on_click
        self._drag_active = False
        self._drag_moved = False
        self._drag_last_pos = QPoint()
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setCursor(Qt.OpenHandCursor)

    def wheelEvent(self, event) -> None:
        if self._on_zoom is None:
            super().wheelEvent(event)
            return
        self._on_zoom(event.angleDelta().y())
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_active = True
            self._drag_moved = False
            self._drag_last_pos = event.globalPos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_active:
            delta = event.globalPos() - self._drag_last_pos
            if delta.manhattanLength() > 0:
                scroll_area = self._parent_scroll_area()
                if scroll_area is not None:
                    scroll_area.horizontalScrollBar().setValue(scroll_area.horizontalScrollBar().value() - delta.x())
                    scroll_area.verticalScrollBar().setValue(scroll_area.verticalScrollBar().value() - delta.y())
                self._drag_moved = self._drag_moved or delta.manhattanLength() > 2
                self._drag_last_pos = event.globalPos()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._drag_active:
            self._drag_active = False
            self.setCursor(Qt.OpenHandCursor)
            if not self._drag_moved and self._on_click is not None:
                self._on_click()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _parent_scroll_area(self) -> QScrollArea | None:
        parent = self.parentWidget()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                return parent
            parent = parent.parentWidget()
        return None


class PreviewLabel(InteractiveImageLabel):
    def __init__(self, text: str, on_zoom=None, on_click=None):
        super().__init__(text, on_zoom=on_zoom, on_click=on_click)
        self.setObjectName("preview-label")
        self.setAlignment(Qt.AlignCenter)
        stage_min_length = _preview_stage_min_length()
        self.setMinimumSize(stage_min_length, stage_min_length)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setProperty("previewClickAction", "open-dialog")


class PreviewDialog(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(TEXT["dialog_title"])
        self.resize(960, 960)
        self.current_pixmap = QPixmap()
        self.zoom_factor = 1.0
        self.base_scale = 1.0
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setStyleSheet("background: #f8f2e4; border: 1px solid #d9cfb8; border-radius: 16px;")
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.preview_image = InteractiveImageLabel(TEXT["dialog_empty"], on_zoom=self._zoom_by_delta)
        self.preview_image.setObjectName("preview-dialog-label")
        self.preview_image.setText(TEXT["dialog_empty"])
        self.scroll_area.setWidget(self.preview_image)
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(self.scroll_area)
        self.setLayout(layout)

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self.current_pixmap = pixmap
        self.zoom_factor = 1.0
        if pixmap.isNull():
            self.preview_image.setPixmap(QPixmap())
            self.preview_image.setText(TEXT["dialog_empty"])
            self.preview_image.resize(0, 0)
            return
        self.preview_image.setText("")
        self._update_display()

    def _zoom_by_delta(self, delta: int) -> None:
        if self.current_pixmap.isNull():
            return
        step = 1.15 if delta > 0 else 1 / 1.15
        self.zoom_factor = max(0.2, min(8.0, self.zoom_factor * step))
        self._update_display()

    def _update_display(self) -> None:
        if self.current_pixmap.isNull():
            return
        viewport = self.scroll_area.viewport().size()
        if viewport.width() <= 0 or viewport.height() <= 0:
            viewport = self.size()
        if self.current_pixmap.width() > 0 and self.current_pixmap.height() > 0:
            self.base_scale = min(
                viewport.width() / self.current_pixmap.width(),
                viewport.height() / self.current_pixmap.height(),
                1.0,
            )
        else:
            self.base_scale = 1.0
        scale = max(0.1, self.base_scale * self.zoom_factor)
        scaled = self.current_pixmap.scaled(
            max(1, int(self.current_pixmap.width() * scale)),
            max(1, int(self.current_pixmap.height() * scale)),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview_image.setPixmap(scaled)
        self.preview_image.resize(scaled.size())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self.current_pixmap.isNull():
            self._update_display()


class VideoDetectionWorker(QObject):
    progress = pyqtSignal(int, object)
    frame_ready = pyqtSignal(object, object)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        detection_service: DetectionService,
        video_path: Path,
        conf: float,
        iou: float,
    ):
        super().__init__()
        self.detection_service = detection_service
        self.video_path = video_path
        self.conf = conf
        self.iou = iou

    def run(self) -> None:
        try:
            result = self.detection_service.detect_video(
                self.video_path,
                conf=self.conf,
                iou=self.iou,
                progress_callback=self._emit_progress,
                frame_callback=self._emit_frame,
            )
        except Exception as exc:  # pragma: no cover - exercised through GUI integration
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)

    def _emit_progress(self, processed: int, total: int | None) -> None:
        self.progress.emit(processed, total)

    def _emit_frame(self, frame_result: VideoFrameResult, total: int | None) -> None:
        self.frame_ready.emit(frame_result, total)


class MainWindow(QMainWindow):
    def __init__(self, detection_service: DetectionService, output_dir: str | Path):
        super().__init__()
        self.detection_service = detection_service
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_result: DetectionResult | None = None
        self.current_image_path: Path | None = None
        self.single_image_path: Path | None = None
        self.single_result: DetectionResult | None = None
        self.current_batch_dir: Path | None = None
        self.current_batch_index: int | None = None
        self.batch_selection_name = "selected_images"
        self.batch_image_paths: list[Path] = []
        self.batch_results: list[DetectionResult] = []
        self.current_mode = "image"
        self.showing_annotated = True
        self.single_showing_annotated = False
        self.batch_showing_annotated = False
        self.original_pixmap = QPixmap()
        self.annotated_pixmap = QPixmap()
        self.current_preview_pixmap = QPixmap()
        self.preview_zoom_factor = 1.0
        self.preview_base_scale = 1.0
        self.single_original_pixmap = QPixmap()
        self.single_annotated_pixmap = QPixmap()
        self.current_video_path: Path | None = None
        self.current_video_result: VideoDetectionResult | None = None
        self.video_original_pixmap = QPixmap()
        self.video_annotated_pixmap = QPixmap()
        self.video_showing_annotated = False
        self.video_processing = False
        self.video_processed_frames = 0
        self.video_total_frames: int | None = None
        self.video_paused = False
        self.video_playback_finished = False
        self.video_current_frame_index: int | None = None
        self.video_next_frame_index = 0
        self.video_current_original_pixmap = QPixmap()
        self.video_current_annotated_pixmap = QPixmap()
        self.video_thread: QThread | None = None
        self.video_worker: VideoDetectionWorker | None = None
        self.video_start_time: float | None = None
        self.video_playback_timer = QTimer(self)
        self.video_playback_timer.setInterval(250)
        self.video_playback_timer.timeout.connect(self._play_next_video_frame)
        self.camera_open = False
        self.camera_detection_active = False
        self.camera_capture = None
        self.camera_current_result: CameraFrameResult | None = None
        self.camera_last_raw_frame = None
        self.camera_last_annotated_frame = None
        self.camera_frame_counter = 0
        self.camera_detection_started_at: float | None = None
        self.camera_frame_timer = QTimer(self)
        self.camera_frame_timer.setInterval(33)
        self.camera_frame_timer.timeout.connect(self._update_camera_preview)
        self.preview_dialog = PreviewDialog(self)
        self.model_name = "yolo26n.pt"

        self.setWindowTitle("Lychee Detection Console")
        self.resize(1500, 1000)

        self._build_ui()
        self._apply_styles()
        self._apply_mode("image")

    def _label(self, text: str, object_name: str | None = None, point_size: int = 18) -> QLabel:
        label = QLabel(text)
        if object_name:
            label.setObjectName(object_name)
        font = QFont()
        font.setPointSize(point_size)
        label.setFont(font)
        return label

    def _build_ui(self) -> None:
        page = QWidget()
        page_layout = QVBoxLayout()
        page_layout.setContentsMargins(28, 24, 28, 24)
        page_layout.setSpacing(18)
        page_layout.addWidget(self._build_mode_header())
        page_layout.addWidget(self._build_status_row())
        page_layout.addWidget(self._build_main_area(), stretch=1)
        page.setLayout(page_layout)
        self.setCentralWidget(page)

    def _build_mode_header(self) -> QWidget:
        container = QWidget()
        container.setObjectName("mode-switcher")
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        self.mode_buttons: dict[str, QPushButton] = {}
        for mode, text in (
            ("image", TEXT["image_mode"]),
            ("video", TEXT["video_mode"]),
            ("camera", TEXT["camera_mode"]),
        ):
            button = QPushButton(text)
            button.setObjectName(f"mode-entry-{mode}")
            button.setCheckable(True)
            button.setMinimumHeight(86)
            button.clicked.connect(lambda checked=False, value=mode: self._apply_mode(value))
            self.mode_buttons[mode] = button
            layout.addWidget(button)
        container.setLayout(layout)
        return container

    def _build_status_row(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("status-info-bar")
        layout = QHBoxLayout()
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(14)
        labels = [
            self._label(f"{TEXT['model']}  {self.model_name}", "status-model-label", 18),
            self._label(f"{TEXT['current_file']}  \u672a\u9009\u62e9", "status-file-label", 18),
            self._label(f"{TEXT['state']}  {TEXT['ready']}", "status-state-label", 18),
            self._label(f"{TEXT['elapsed']}  0.00s", "status-time-label", 18),
        ]
        self.status_model_label, self.status_file_label, self.status_state_label, self.status_time_label = labels
        for label in labels:
            card = QFrame()
            card.setObjectName("status-metric-card")
            card_layout = QHBoxLayout()
            card_layout.setContentsMargins(20, 12, 20, 12)
            card_layout.addWidget(label)
            card.setLayout(card_layout)
            layout.addWidget(card, 1)
        bar.setLayout(layout)
        return bar

    def _build_main_area(self) -> QWidget:
        container = QWidget()
        container.setObjectName("main-two-column")
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        layout.addWidget(self._build_left_column(), 2)
        layout.addWidget(self._build_right_column(), 6)
        container.setLayout(layout)
        return container

    def _build_left_column(self) -> QWidget:
        container = QWidget()
        container.setObjectName("left-column-stack")
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        layout.addWidget(self._build_config_panel(), 2)
        layout.addWidget(self._build_result_panel(), 3)
        container.setLayout(layout)
        return container

    def _build_right_column(self) -> QWidget:
        container = QWidget()
        container.setObjectName("right-preview-column")
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_preview_panel())
        container.setLayout(layout)
        return container

    def _build_config_panel(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("left-config-panel")
        frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        self.config_title_label = self._label(TEXT["image_config"], "section-title", 26)
        layout.addWidget(self.config_title_label)

        self.mode_stack = QStackedWidget()
        self.mode_stack.setObjectName("mode-config-stack")
        self.mode_stack.addWidget(self._build_image_config_panel())
        self.mode_stack.addWidget(self._build_video_config_panel())
        self.mode_stack.addWidget(self._build_camera_config_panel())
        layout.addWidget(self.mode_stack)
        frame.setLayout(layout)
        return frame

    def _build_image_config_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("image-config-panel")
        layout = QVBoxLayout()
        layout.setSpacing(18)

        self.config_source_label = self._label(TEXT["image_source"], "config-source-label", 18)
        self.image_source_selector = QComboBox()
        self.image_source_selector.addItems([TEXT["single_image"], TEXT["batch_folder"]])
        self.image_source_selector.setMinimumHeight(50)
        self.image_source_selector.currentIndexChanged.connect(self._handle_source_change)

        self.config_conf_label = self._label(TEXT["confidence"], "config-confidence-label", 18)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setObjectName("config-confidence-slider")
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(25)
        self.conf_slider.setMinimumHeight(34)
        self.conf_value_label = self._label("0.25", "config-confidence-value", 18)
        self.conf_value_label.setMinimumWidth(64)
        self.conf_slider.valueChanged.connect(self._sync_slider_values)

        self.config_iou_label = self._label(TEXT["iou"], "config-iou-label", 18)
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setObjectName("config-iou-slider")
        self.iou_slider.setRange(1, 100)
        self.iou_slider.setValue(45)
        self.iou_slider.setMinimumHeight(34)
        self.iou_value_label = self._label("0.45", "config-iou-value", 18)
        self.iou_value_label.setMinimumWidth(64)
        self.iou_slider.valueChanged.connect(self._sync_slider_values)

        self.config_label_toggle_label = self._label(TEXT["label_visibility"], None, 18)
        self.label_toggle = QCheckBox(TEXT["show_labels"])
        self.label_toggle.setObjectName("config-show-labels-checkbox")
        self.label_toggle.setChecked(True)
        self.label_toggle.setMinimumHeight(34)

        self.config_export_label = self._label(TEXT["export_format"], None, 18)
        self.export_format_selector = QComboBox()
        self.export_format_selector.setObjectName("config-export-format")
        self.export_format_selector.addItems([TEXT["export_both"], TEXT["export_csv_only"], TEXT["export_image_only"]])
        self.export_format_selector.setMinimumHeight(50)

        layout.addWidget(self.config_source_label)
        layout.addWidget(self.image_source_selector)
        layout.addWidget(self.config_conf_label)
        layout.addLayout(self._build_slider_row(self.conf_slider, self.conf_value_label))
        layout.addWidget(self.config_iou_label)
        layout.addLayout(self._build_slider_row(self.iou_slider, self.iou_value_label))
        layout.addWidget(self.config_label_toggle_label)
        layout.addWidget(self.label_toggle)
        layout.addWidget(self.config_export_label)
        layout.addWidget(self.export_format_selector)
        panel.setLayout(layout)
        return panel

    def _build_video_config_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("video-config-panel")
        layout = QVBoxLayout()
        layout.setSpacing(18)

        self.video_source_label = self._label(TEXT["video_source"], "video-source-label", 18)
        self.video_source_value = self._label(TEXT["need_video_body"], "video-source-value", 16)
        self.video_source_value.setWordWrap(True)

        self.video_conf_label = self._label(TEXT["confidence"], "video-confidence-label", 18)
        self.video_conf_slider = QSlider(Qt.Horizontal)
        self.video_conf_slider.setObjectName("video-confidence-slider")
        self.video_conf_slider.setRange(1, 100)
        self.video_conf_slider.setValue(25)
        self.video_conf_slider.setMinimumHeight(34)
        self.video_conf_value_label = self._label("0.25", "video-confidence-value", 18)
        self.video_conf_slider.valueChanged.connect(self._sync_slider_values)

        self.video_iou_label = self._label(TEXT["iou"], "video-iou-label", 18)
        self.video_iou_slider = QSlider(Qt.Horizontal)
        self.video_iou_slider.setObjectName("video-iou-slider")
        self.video_iou_slider.setRange(1, 100)
        self.video_iou_slider.setValue(45)
        self.video_iou_slider.setMinimumHeight(34)
        self.video_iou_value_label = self._label("0.45", "video-iou-value", 18)
        self.video_iou_slider.valueChanged.connect(self._sync_slider_values)

        self.video_label_toggle_label = self._label(TEXT["label_visibility"], "video-label-visibility-label", 18)
        self.video_label_toggle = QCheckBox(TEXT["show_labels"])
        self.video_label_toggle.setObjectName("video-show-labels-checkbox")
        self.video_label_toggle.setChecked(True)
        self.video_label_toggle.setMinimumHeight(34)

        self.video_export_label = self._label(TEXT["export_format"], "video-export-label", 18)
        self.video_export_format_selector = QComboBox()
        self.video_export_format_selector.setObjectName("video-export-format")
        self.video_export_format_selector.addItems(
            [TEXT["export_both"], TEXT["export_csv_only"], TEXT["export_video_only"]]
        )
        self.video_export_format_selector.setMinimumHeight(50)

        self.video_status_label = self._label(TEXT["video_status_idle"], "video-status-label", 17)
        self.video_status_label.setWordWrap(True)

        layout.addWidget(self.video_source_label)
        layout.addWidget(self.video_source_value)
        layout.addWidget(self.video_conf_label)
        layout.addLayout(self._build_slider_row(self.video_conf_slider, self.video_conf_value_label))
        layout.addWidget(self.video_iou_label)
        layout.addLayout(self._build_slider_row(self.video_iou_slider, self.video_iou_value_label))
        layout.addWidget(self.video_label_toggle_label)
        layout.addWidget(self.video_label_toggle)
        layout.addWidget(self.video_export_label)
        layout.addWidget(self.video_export_format_selector)
        layout.addWidget(self.video_status_label)
        layout.addStretch(1)
        panel.setLayout(layout)
        return panel

    def _build_camera_config_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("camera-config-panel")
        layout = QVBoxLayout()
        layout.setSpacing(18)

        self.camera_device_label = self._label(TEXT["camera_device"], "camera-device-label", 18)
        self.camera_device_selector = QComboBox()
        self.camera_device_selector.setObjectName("camera-device-selector")
        self.camera_device_selector.addItem(TEXT["camera_device_default"], 0)
        self.camera_device_selector.setMinimumHeight(50)

        self.camera_conf_label = self._label(TEXT["confidence"], "camera-confidence-label", 18)
        self.camera_conf_slider = QSlider(Qt.Horizontal)
        self.camera_conf_slider.setObjectName("camera-confidence-slider")
        self.camera_conf_slider.setRange(1, 100)
        self.camera_conf_slider.setValue(25)
        self.camera_conf_slider.setMinimumHeight(34)
        self.camera_conf_value_label = self._label("0.25", "camera-confidence-value", 16)
        self.camera_conf_slider.valueChanged.connect(self._sync_slider_values)

        self.camera_iou_label = self._label(TEXT["iou"], "camera-iou-label", 18)
        self.camera_iou_slider = QSlider(Qt.Horizontal)
        self.camera_iou_slider.setObjectName("camera-iou-slider")
        self.camera_iou_slider.setRange(1, 100)
        self.camera_iou_slider.setValue(45)
        self.camera_iou_slider.setMinimumHeight(34)
        self.camera_iou_value_label = self._label("0.45", "camera-iou-value", 16)
        self.camera_iou_slider.valueChanged.connect(self._sync_slider_values)

        self.camera_label_toggle_label = self._label(TEXT["label_visibility"], "camera-label-visibility-label", 18)
        self.camera_label_toggle = QCheckBox(TEXT["show_labels"])
        self.camera_label_toggle.setObjectName("camera-show-labels-checkbox")
        self.camera_label_toggle.setChecked(True)
        self.camera_label_toggle.setMinimumHeight(34)

        self.camera_save_frames_checkbox = QCheckBox(TEXT["camera_save_frames"])
        self.camera_save_frames_checkbox.setObjectName("camera-save-frames-checkbox")
        self.camera_save_frames_checkbox.setMinimumHeight(34)

        self.camera_status_label = self._label(TEXT["camera_status_ready"], "camera-status-label", 17)
        self.camera_status_label.setWordWrap(True)

        layout.addWidget(self.camera_device_label)
        layout.addWidget(self.camera_device_selector)
        layout.addWidget(self.camera_conf_label)
        layout.addLayout(self._build_slider_row(self.camera_conf_slider, self.camera_conf_value_label))
        layout.addWidget(self.camera_iou_label)
        layout.addLayout(self._build_slider_row(self.camera_iou_slider, self.camera_iou_value_label))
        layout.addWidget(self.camera_label_toggle_label)
        layout.addWidget(self.camera_label_toggle)
        layout.addWidget(self.camera_save_frames_checkbox)
        layout.addWidget(self.camera_status_label)
        layout.addStretch(1)
        panel.setLayout(layout)
        return panel

    def _build_slider_row(self, slider: QSlider, value_label: QLabel) -> QHBoxLayout:
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(slider, 1)
        layout.addWidget(value_label)
        return layout

    def _build_placeholder(self, text: str, object_name: str) -> QWidget:
        panel = QWidget()
        panel.setObjectName(object_name)
        layout = QVBoxLayout()
        layout.addWidget(self._label(text, None, 18))
        layout.addStretch(1)
        panel.setLayout(layout)
        return panel

    def _build_preview_panel(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("preview-panel")
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 10, 12, 8)
        layout.setSpacing(2)

        self.preview_action_bar = QWidget()
        self.preview_action_bar.setObjectName("preview-action-bar")
        self.preview_action_center = QWidget()
        self.preview_action_center.setObjectName("preview-action-center")
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(16)
        button_layout.setAlignment(Qt.AlignHCenter)

        open_button = QPushButton(TEXT["open"])
        open_button.setObjectName("preview-action-open")
        open_button.setMinimumHeight(56)
        open_button.clicked.connect(self.open_current_source)
        detect_button = QPushButton(TEXT["detect"])
        detect_button.setObjectName("preview-action-detect")
        detect_button.setMinimumHeight(56)
        detect_button.clicked.connect(self.run_detection)
        preview_button = QPushButton(TEXT["preview"])
        preview_button.setObjectName("preview-action-preview")
        preview_button.setMinimumHeight(56)
        preview_button.clicked.connect(self.open_preview_dialog)
        self.video_original_button = QPushButton(TEXT["video_original"])
        self.video_original_button.setObjectName("preview-action-video-original")
        self.video_original_button.setMinimumHeight(56)
        self.video_original_button.clicked.connect(self.toggle_video_preview_frame)
        self.video_result_button = QPushButton(TEXT["video_result"])
        self.video_result_button.setObjectName("preview-action-video-result")
        self.video_result_button.setMinimumHeight(56)
        self.video_result_button.clicked.connect(self.show_video_annotated_frame)
        self.video_restart_button = QPushButton(TEXT["video_restart"])
        self.video_restart_button.setObjectName("preview-action-video-restart")
        self.video_restart_button.setMinimumHeight(56)
        self.video_restart_button.clicked.connect(self.restart_video_playback)
        export_button = QPushButton(TEXT["export"])
        export_button.setObjectName("preview-action-export")
        export_button.setMinimumHeight(56)
        export_button.clicked.connect(self.export_by_selection)
        self.close_current_button = QPushButton(TEXT["close_current"])
        self.close_current_button.setObjectName("preview-action-close-current")
        self.close_current_button.setMinimumHeight(56)
        self.close_current_button.setMinimumWidth(56)
        self.close_current_button.setMaximumWidth(56)
        self.close_current_button.clicked.connect(self.close_current_display)
        self.close_all_button = QPushButton(TEXT["close_all"])
        self.close_all_button.setObjectName("preview-action-close-all")
        self.close_all_button.setMinimumHeight(56)
        self.close_all_button.clicked.connect(self.close_all_displays)

        self.preview_action_buttons = [
            open_button,
            detect_button,
            preview_button,
            self.video_original_button,
            self.video_result_button,
            self.video_restart_button,
            export_button,
            self.close_current_button,
            self.close_all_button,
        ]
        self.open_action_button = open_button
        self.detect_action_button = detect_button
        self.preview_action_button = preview_button
        self.export_action_button = export_button
        self.preview_hint_label = None
        for button in self.preview_action_buttons:
            button_layout.addWidget(button)

        self.preview_action_center.setLayout(button_layout)
        action_layout = QHBoxLayout()
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.addStretch(1)
        action_layout.addWidget(self.preview_action_center)
        action_layout.addStretch(1)
        self.preview_action_bar.setLayout(action_layout)
        layout.addWidget(self.preview_action_bar)

        self.preview_nav_prev = QPushButton(TEXT["preview_previous"])
        self.preview_nav_prev.setObjectName("preview-nav-prev")
        self.preview_nav_prev.setMinimumSize(64, 64)
        self.preview_nav_prev.setMaximumWidth(64)
        self.preview_nav_prev.clicked.connect(self.show_previous_batch_image)
        self.preview_nav_next = QPushButton(TEXT["preview_next"])
        self.preview_nav_next.setObjectName("preview-nav-next")
        self.preview_nav_next.setMinimumSize(64, 64)
        self.preview_nav_next.setMaximumWidth(64)
        self.preview_nav_next.clicked.connect(self.show_next_batch_image)

        self.preview_scroll_area = QScrollArea()
        self.preview_scroll_area.setObjectName("preview-scroll-area")
        self.preview_scroll_area.setWidgetResizable(False)
        self.preview_scroll_area.setFrameShape(QFrame.NoFrame)
        self.preview_scroll_area.setAlignment(Qt.AlignCenter)
        self.preview_scroll_area.setStyleSheet("background: transparent;")
        stage_min_length = _preview_stage_min_length()
        self.preview_scroll_area.setMinimumSize(stage_min_length, stage_min_length)

        self.image_label = PreviewLabel(
            TEXT["preview_empty"],
            on_zoom=self._zoom_preview_by_delta,
            on_click=lambda: self._handle_preview_click(None),
        )
        self.preview_scroll_area.setWidget(self.image_label)
        preview_stage_layout = QHBoxLayout()
        preview_stage_layout.setContentsMargins(0, 0, 0, 0)
        preview_stage_layout.setSpacing(14)
        preview_stage_layout.addWidget(self.preview_nav_prev, alignment=Qt.AlignVCenter)
        preview_stage_layout.addWidget(self.preview_scroll_area, alignment=Qt.AlignCenter, stretch=1)
        preview_stage_layout.addWidget(self.preview_nav_next, alignment=Qt.AlignVCenter)
        layout.addLayout(preview_stage_layout, 1)

        hint = self._label(TEXT["preview_hint"], "preview-hint-label", 14)
        hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(hint)
        self.preview_hint_label = hint
        self._update_batch_navigation()
        frame.setLayout(layout)
        return frame

    def _build_result_panel(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("left-result-panel")
        layout = QVBoxLayout()
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        self.result_overview_card = QFrame()
        self.result_overview_card.setObjectName("result-overview-card")
        overview_layout = QVBoxLayout()
        overview_layout.setContentsMargins(14, 12, 14, 12)
        overview_layout.setSpacing(6)
        overview_layout.addWidget(self._label(TEXT["summary_title"], "section-title", 22))
        self.overview_value_label = self._label(TEXT["no_result"], "result-overview-value", 17)
        self.overview_value_label.setWordWrap(True)
        overview_layout.addWidget(self.overview_value_label)
        self.result_overview_card.setLayout(overview_layout)

        self.result_detail_card = QFrame()
        self.result_detail_card.setObjectName("result-detail-card")
        detail_layout = QVBoxLayout()
        detail_layout.setContentsMargins(14, 12, 14, 12)
        detail_layout.setSpacing(6)
        detail_layout.addWidget(self._label(TEXT["detail_title"], "section-title", 22))
        self.detail_list = QListWidget()
        self.detail_list.setObjectName("result-detail-list")
        self.detail_list.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        detail_layout.addWidget(self.detail_list)
        self.result_detail_card.setLayout(detail_layout)

        layout.addWidget(self.result_overview_card)
        layout.addWidget(self.result_detail_card, 1)
        frame.setLayout(layout)
        return frame

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background: #ece5d6; }
            QWidget { color: #2f3a2c; font-size: 20px; }
            QFrame#status-info-bar,
            QFrame#left-config-panel,
            QFrame#left-result-panel,
            QFrame#preview-panel {
                background: #f7f1e3;
                border: 1px solid #d9cfb8;
                border-radius: 22px;
            }
            QFrame#status-metric-card,
            QFrame#result-overview-card,
            QFrame#result-detail-card {
                background: #fffdf8;
                border: 1px solid #dfd4be;
                border-radius: 16px;
            }
            QLabel#section-title {
                font-size: 28px;
                font-weight: 700;
                color: #304127;
            }
            QLabel#preview-hint-label {
                color: #7e765f;
                font-size: 15px;
            }
            QPushButton {
                background: #f1e0a6;
                border: 1px solid #ceb977;
                border-radius: 16px;
                padding: 12px 26px;
                min-width: 126px;
                font-size: 22px;
            }
            QPushButton:hover { background: #ead48f; }
            QPushButton:checked {
                background: #afc26f;
                border-color: #8aa051;
                color: #1f2a1a;
            }
            QPushButton#preview-action-close-current {
                min-width: 56px;
                max-width: 56px;
                padding: 12px 0;
                font-size: 24px;
            }
            QComboBox, QListWidget {
                background: #fffdf8;
                border: 1px solid #d9cfb8;
                border-radius: 14px;
                padding: 8px 12px;
                font-size: 20px;
            }
            QSlider::groove:horizontal {
                background: #e7dcc8;
                height: 10px;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: #7e9a4d;
                width: 22px;
                margin: -6px 0;
                border-radius: 11px;
            }
            QCheckBox { spacing: 12px; font-size: 20px; }
            QCheckBox::indicator { width: 24px; height: 24px; }
            """
        )

    def _apply_mode(self, mode: str) -> None:
        if self.video_processing and mode != self.current_mode:
            QMessageBox.information(self, TEXT["video_running_block_title"], TEXT["video_running_block_body"])
            self.mode_buttons[self.current_mode].setChecked(True)
            return
        if self.current_mode == "image" and mode != "image":
            self._remember_current_single_state()
        if self.current_mode == "camera" and mode != "camera":
            self._stop_camera_preview(clear_display=False)
        self.current_mode = mode
        self.mode_stack.setCurrentIndex({"image": 0, "video": 1, "camera": 2}[mode])
        for name, button in self.mode_buttons.items():
            button.setChecked(name == mode)
        if mode == "image":
            self.config_title_label.setText(TEXT["image_config"])
            self.camera_open = False
            self.camera_detection_active = False
            self.status_state_label.setText(f"{TEXT['state']}  {TEXT['ready']}")
            self._restore_source_state()
        elif mode == "video":
            self.config_title_label.setText(TEXT["video_config"])
            self.camera_open = False
            self.camera_detection_active = False
            self.status_state_label.setText(f"{TEXT['state']}  {TEXT['video_ready']}")
            self._restore_video_state()
        else:
            self.config_title_label.setText(TEXT["camera_mode"])
            self.camera_open = False
            self.camera_detection_active = False
            self._clear_visible_display()
            self.status_file_label.setText(f"{TEXT['current_file']}  {TEXT['camera_device_default']}")
            self.status_state_label.setText(f"{TEXT['state']}  \u6444\u50cf\u5934\u5df2\u5c31\u7eea")
            self.overview_value_label.setText(TEXT["camera_overview_ready"])
            self.camera_status_label.setText(TEXT["camera_status_ready"])
            self.detail_list.clear()
        self._update_batch_navigation()
        self._update_preview_action_visibility()

    def _sync_slider_values(self) -> None:
        self.conf_value_label.setText(f"{self.conf_slider.value() / 100:.2f}")
        self.iou_value_label.setText(f"{self.iou_slider.value() / 100:.2f}")
        if hasattr(self, "video_conf_value_label"):
            self.video_conf_value_label.setText(f"{self.video_conf_slider.value() / 100:.2f}")
            self.video_iou_value_label.setText(f"{self.video_iou_slider.value() / 100:.2f}")
        if hasattr(self, "camera_conf_value_label"):
            self.camera_conf_value_label.setText(f"{self.camera_conf_slider.value() / 100:.2f}")
            self.camera_iou_value_label.setText(f"{self.camera_iou_slider.value() / 100:.2f}")

    def _handle_source_change(self, index: int) -> None:
        if index == 1:
            self._remember_current_single_state(force=True)
        else:
            self._remember_current_single_state()
        self._restore_source_state()
        self._update_preview_action_visibility()

    def _reset_selection_state(self, clear_source: bool = True) -> None:
        self.current_result = None
        self.current_image_path = None
        self.single_image_path = None
        self.single_result = None
        self.current_batch_dir = None
        self.current_batch_index = None
        self.batch_selection_name = "selected_images"
        self.batch_image_paths = []
        self.batch_results = []
        self.original_pixmap = QPixmap()
        self.annotated_pixmap = QPixmap()
        self.current_preview_pixmap = QPixmap()
        self.single_original_pixmap = QPixmap()
        self.single_annotated_pixmap = QPixmap()
        self.current_video_path = None
        self.current_video_result = None
        self.video_original_pixmap = QPixmap()
        self.video_annotated_pixmap = QPixmap()
        self.video_showing_annotated = False
        self.video_processing = False
        self.video_processed_frames = 0
        self.video_total_frames = None
        self.video_paused = False
        self.video_playback_finished = False
        self.video_current_frame_index = None
        self.video_next_frame_index = 0
        self.video_current_original_pixmap = QPixmap()
        self.video_current_annotated_pixmap = QPixmap()
        self.video_playback_timer.stop()
        self.showing_annotated = False
        self.single_showing_annotated = False
        self.batch_showing_annotated = False
        self._clear_visible_display()
        self._update_batch_navigation()
        self.detail_list.clear()
        self.status_file_label.setText(f"{TEXT['current_file']}  \u672a\u9009\u62e9")
        self.status_time_label.setText(f"{TEXT['elapsed']}  0.00s")
        self.overview_value_label.setText(TEXT["no_result"])
        if clear_source and hasattr(self, "image_source_selector"):
            self.image_source_selector.blockSignals(True)
            self.image_source_selector.setCurrentIndex(0)
            self.image_source_selector.blockSignals(False)
        self._update_preview_action_visibility()

    def _current_confidence(self) -> float:
        if self.current_mode == "camera":
            return self.camera_conf_slider.value() / 100
        if self.current_mode == "video":
            return self.video_conf_slider.value() / 100
        return self.conf_slider.value() / 100

    def _current_iou(self) -> float:
        if self.current_mode == "camera":
            return self.camera_iou_slider.value() / 100
        if self.current_mode == "video":
            return self.video_iou_slider.value() / 100
        return self.iou_slider.value() / 100

    def _scan_batch_images(self, folder_path: Path) -> list[Path]:
        supported = {".png", ".jpg", ".jpeg", ".bmp"}
        return sorted(
            [
                path
                for path in folder_path.iterdir()
                if path.is_file() and path.suffix.lower() in supported
            ]
        )

    def _handle_preview_click(self, _event) -> None:
        if self.current_mode == "video":
            self._toggle_video_pause()
            return
        self.toggle_preview_mode()

    def _zoom_preview_by_delta(self, delta: int) -> None:
        if self.current_preview_pixmap.isNull():
            return
        step = 1.15 if delta > 0 else 1 / 1.15
        self.preview_zoom_factor = max(0.2, min(8.0, self.preview_zoom_factor * step))
        self._update_main_preview_display()

    def open_current_source(self) -> None:
        if self.video_processing:
            QMessageBox.information(self, TEXT["video_running_block_title"], TEXT["video_running_block_body"])
            return
        if self.current_mode == "camera":
            self.open_camera()
            return
        if self.current_mode == "video":
            self.select_video()
            return
        self.select_image()

    def open_camera(self) -> None:
        if self.camera_open:
            self._stop_camera_preview(clear_display=True)
            self.status_state_label.setText(f"{TEXT['state']}  \u6444\u50cf\u5934\u5df2\u5173\u95ed")
            self.camera_status_label.setText(TEXT["camera_status_ready"])
            self.overview_value_label.setText(TEXT["camera_overview_ready"])
            self.status_file_label.setText(f"{TEXT['current_file']}  {TEXT['camera_device_default']}")
            self.status_time_label.setText(f"{TEXT['elapsed']}  0.00s")
            self._update_preview_action_visibility()
            return

        if not self._start_camera_preview():
            return

        self.camera_open = True
        self.status_state_label.setText(f"{TEXT['state']}  \u6444\u50cf\u5934\u5df2\u6253\u5f00")
        self.camera_status_label.setText(TEXT["camera_status_opened"])
        self.overview_value_label.setText(TEXT["camera_overview_ready"])
        self.status_file_label.setText(f"{TEXT['current_file']}  {self.camera_device_selector.currentText()}")
        self.status_time_label.setText(f"{TEXT['elapsed']}  0.00s")
        self._update_preview_action_visibility()

    def _selected_camera_index(self) -> int:
        current_data = self.camera_device_selector.currentData()
        return int(current_data if current_data is not None else 0)

    def _camera_backend_candidates(self) -> list[int | None]:
        if cv2 is None:
            return [None]
        if not sys.platform.startswith("win"):
            return [None]
        candidates: list[int | None] = []
        for backend_name in ("CAP_DSHOW", "CAP_MSMF"):
            backend = getattr(cv2, backend_name, None)
            if backend is not None and backend not in candidates:
                candidates.append(backend)
        candidates.append(None)
        return candidates

    def _is_valid_camera_frame(self, frame) -> bool:
        if frame is None or not hasattr(frame, "size") or frame.size == 0:
            return False
        if getattr(frame, "ndim", 0) not in (2, 3):
            return False
        if frame.ndim == 3 and frame.shape[2] not in (3, 4):
            return False
        return bool(frame.max() > 0)

    def _open_camera_capture(self):
        camera_index = self._selected_camera_index()
        warmup_frames = 6
        for backend in self._camera_backend_candidates():
            capture = cv2.VideoCapture(camera_index) if backend is None else cv2.VideoCapture(camera_index, backend)
            if not capture.isOpened():
                capture.release()
                continue
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            last_valid_frame = None
            for _ in range(warmup_frames):
                ok, frame = capture.read()
                if ok and self._is_valid_camera_frame(frame):
                    last_valid_frame = frame
            if last_valid_frame is not None:
                return capture, last_valid_frame
            capture.release()
        return None, None

    def _start_camera_preview(self) -> bool:
        if cv2 is None:
            QMessageBox.warning(self, TEXT["camera_unavailable_title"], TEXT["camera_unavailable_body"])
            return False

        capture, first_frame = self._open_camera_capture()
        if capture is None or first_frame is None:
            QMessageBox.warning(self, TEXT["camera_unavailable_title"], TEXT["camera_busy_body"])
            return False

        self.camera_capture = capture
        self.camera_current_result = None
        self.camera_last_raw_frame = first_frame.copy()
        self.camera_last_annotated_frame = None
        raw_pixmap = self._camera_frame_to_qpixmap(first_frame)
        self.original_pixmap = raw_pixmap
        self.annotated_pixmap = QPixmap()
        self.current_preview_pixmap = raw_pixmap
        self.showing_annotated = False
        self.detail_list.clear()
        self._set_preview(raw_pixmap)
        self.preview_dialog.set_pixmap(raw_pixmap)
        self.camera_frame_timer.start()
        return True

    def _stop_camera_preview(self, clear_display: bool) -> None:
        self.camera_detection_active = False
        self.camera_open = False
        self.camera_detection_started_at = None
        self.camera_frame_timer.stop()
        if self.camera_capture is not None:
            self.camera_capture.release()
            self.camera_capture = None
        self.camera_current_result = None
        self.camera_last_raw_frame = None
        self.camera_last_annotated_frame = None
        self.camera_frame_counter = 0
        if clear_display:
            self.current_preview_pixmap = QPixmap()
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText(TEXT["preview_empty"])
            self.preview_dialog.set_pixmap(QPixmap())
            self.detail_list.clear()

    def _update_camera_preview(self) -> None:
        if self.camera_capture is None:
            return

        ok, frame = self.camera_capture.read()
        if not ok or not self._is_valid_camera_frame(frame):
            self.camera_status_label.setText(TEXT["camera_status_preview_error"])
            return

        self.camera_last_raw_frame = frame.copy()
        raw_pixmap = self._camera_frame_to_qpixmap(frame)
        self.original_pixmap = raw_pixmap

        if self.camera_detection_active:
            self._run_camera_frame_detection(frame)
            return

        self.camera_current_result = None
        self.camera_last_annotated_frame = None
        self.annotated_pixmap = QPixmap()
        self.current_preview_pixmap = raw_pixmap
        self.showing_annotated = False
        self.detail_list.clear()
        self._set_preview(raw_pixmap)
        self.preview_dialog.set_pixmap(raw_pixmap)

    def _run_camera_frame_detection(self, frame) -> None:
        started = perf_counter()
        try:
            frame_result = self.detection_service.detect_frame(
                frame,
                conf=self._current_confidence(),
                iou=self._current_iou(),
            )
        except Exception as exc:
            self.camera_detection_active = False
            self.camera_status_label.setText(str(exc))
            self.status_state_label.setText(f"{TEXT['state']}  {TEXT['failed']}")
            self._update_preview_action_visibility()
            return

        annotated_frame = self._draw_camera_detections(frame, frame_result)
        self.camera_current_result = CameraFrameResult(
            detections=frame_result.detections,
            counts_by_label=frame_result.counts_by_label,
            total_detections=frame_result.total_detections,
            annotated_frame=annotated_frame,
        )
        self.camera_last_annotated_frame = annotated_frame.copy()
        annotated_pixmap = self._camera_frame_to_qpixmap(annotated_frame)
        self.annotated_pixmap = annotated_pixmap
        self.current_preview_pixmap = annotated_pixmap
        self.showing_annotated = True
        self._set_preview(annotated_pixmap)
        self.preview_dialog.set_pixmap(annotated_pixmap)
        self.camera_frame_counter += 1
        if self.camera_detection_started_at is None:
            self.camera_detection_started_at = started
        elapsed = max(perf_counter() - self.camera_detection_started_at, 0.01)
        self.status_time_label.setText(f"{TEXT['elapsed']}  {elapsed:.2f}s")
        self._populate_camera_result_view(self.camera_current_result)
        if self.camera_save_frames_checkbox.isChecked():
            self._save_camera_frame(annotated_frame)

    def _draw_camera_detections(self, frame, frame_result: CameraFrameResult):
        annotated = frame.copy()
        show_labels = self.camera_label_toggle.isChecked()
        for detection in frame_result.detections:
            x1, y1, x2, y2 = (int(value) for value in detection.bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (66, 190, 101), 2)
            if not show_labels:
                continue
            label_text = f"{detection.label} {detection.confidence:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            text_top = max(0, y1 - text_height - baseline - 8)
            text_bottom = text_top + text_height + baseline + 8
            cv2.rectangle(annotated, (x1, text_top), (x1 + text_width + 12, text_bottom), (66, 190, 101), -1)
            cv2.putText(
                annotated,
                label_text,
                (x1 + 6, text_bottom - baseline - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (22, 40, 22),
                2,
                cv2.LINE_AA,
            )
        return annotated

    def _populate_camera_result_view(self, frame_result: CameraFrameResult) -> None:
        counts = "\u3001".join(f"{label} {count}" for label, count in sorted(frame_result.counts_by_label.items()))
        overview_lines = [TEXT["camera_overview_total"].format(count=frame_result.total_detections)]
        if counts:
            overview_lines.append(counts)
        else:
            overview_lines.append(TEXT["overview_none"])
        if self.camera_save_frames_checkbox.isChecked():
            overview_lines.append(TEXT["camera_overview_saved"].format(path=self._camera_frames_dir()))
        self.overview_value_label.setText("\n".join(overview_lines))
        self.detail_list.clear()
        for detection in frame_result.detections:
            bbox_text = ", ".join(f"{value:.1f}" for value in detection.bbox)
            self.detail_list.addItem(
                TEXT["detail_line"].format(
                    label=detection.label,
                    confidence=detection.confidence,
                    bbox=bbox_text,
                )
            )

    def _camera_frames_dir(self) -> Path:
        return self.output_dir / "camera_frames" / f"device_{self._selected_camera_index()}"

    def _save_camera_frame(self, frame) -> None:
        output_dir = self._camera_frames_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"frame_{self.camera_frame_counter:06d}.jpg"
        cv2.imwrite(str(output_path), frame)

    def _camera_frame_to_qpixmap(self, frame) -> QPixmap:
        if frame.ndim == 2:
            height, width = frame.shape
            qimage = QImage(frame.tobytes(), width, height, width, QImage.Format_Grayscale8)
            return QPixmap.fromImage(qimage.copy())
        if frame.shape[2] == 4:
            rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
            height, width, channels = rgba_frame.shape
            bytes_per_line = channels * width
            qimage = QImage(rgba_frame.tobytes(), width, height, bytes_per_line, QImage.Format_RGBA8888)
            return QPixmap.fromImage(qimage.copy())
        height, width, channels = frame.shape
        bytes_per_line = channels * width
        qimage = QImage(frame.tobytes(), width, height, bytes_per_line, QImage.Format_BGR888)
        return QPixmap.fromImage(qimage.copy())

    def open_preview_dialog(self) -> None:
        if self.current_mode == "camera":
            if self.current_preview_pixmap.isNull():
                QMessageBox.information(self, TEXT["preview_missing_title"], TEXT["camera_preview_body"])
                return
            self.preview_dialog.set_pixmap(self.current_preview_pixmap)
            self.preview_dialog.showMaximized()
            self.preview_dialog.raise_()
            self.preview_dialog.activateWindow()
            return
        pixmap = self.annotated_pixmap if self.showing_annotated and not self.annotated_pixmap.isNull() else self.original_pixmap
        if pixmap.isNull():
            QMessageBox.information(self, TEXT["preview_missing_title"], TEXT["preview_missing_body"])
            return
        self.preview_dialog.set_pixmap(pixmap)
        self.preview_dialog.showMaximized()
        self.preview_dialog.raise_()
        self.preview_dialog.activateWindow()

    def _batch_display_name(self) -> str:
        if self.current_batch_dir is not None:
            return self.current_batch_dir.name
        return self.batch_selection_name

    def _update_batch_navigation(self) -> None:
        is_batch = (
            hasattr(self, "image_source_selector")
            and self.current_mode == "image"
            and len(self.batch_image_paths) > 1
            and self.image_source_selector.currentIndex() == 1
        )
        current_index = self.current_batch_index if self.current_batch_index is not None else 0
        self.preview_nav_prev.setVisible(is_batch)
        self.preview_nav_next.setVisible(is_batch)
        self.preview_nav_prev.setEnabled(is_batch and current_index > 0)
        self.preview_nav_next.setEnabled(is_batch and current_index < len(self.batch_image_paths) - 1)

    def _show_batch_item(self, index: int, prefer_annotated: bool | None = None) -> None:
        if not self.batch_image_paths:
            return
        bounded_index = max(0, min(index, len(self.batch_image_paths) - 1))
        image_path = self.batch_image_paths[bounded_index]
        result = self.batch_results[bounded_index] if bounded_index < len(self.batch_results) else None

        self.current_batch_index = bounded_index
        self.current_image_path = image_path
        self.current_result = result
        self.original_pixmap = QPixmap(str(image_path))
        self.annotated_pixmap = _to_qpixmap(result.annotated_image) if result is not None else QPixmap()

        if prefer_annotated is None:
            should_show_annotated = self.batch_showing_annotated and not self.annotated_pixmap.isNull()
        else:
            should_show_annotated = prefer_annotated and not self.annotated_pixmap.isNull()
        self.showing_annotated = should_show_annotated
        self.batch_showing_annotated = should_show_annotated

        self.status_file_label.setText(
            f"{TEXT['current_file']}  {image_path.name} ({bounded_index + 1}/{len(self.batch_image_paths)})"
        )
        pixmap = self.annotated_pixmap if self.showing_annotated else self.original_pixmap
        self._set_preview(pixmap)
        self.preview_dialog.set_pixmap(pixmap)
        if self.batch_results:
            self._populate_batch_result_view(self.batch_results)
        else:
            self.overview_value_label.setText(TEXT["batch_loaded"].format(count=len(self.batch_image_paths)))
            self.detail_list.clear()
        self._update_batch_navigation()
        self._update_preview_action_visibility()

    def show_previous_batch_image(self) -> None:
        if self.current_batch_index is None:
            return
        self._show_batch_item(self.current_batch_index - 1)

    def show_next_batch_image(self) -> None:
        if self.current_batch_index is None:
            return
        self._show_batch_item(self.current_batch_index + 1)

    def select_image(self) -> None:
        if self.current_mode != "image":
            QMessageBox.information(self, TEXT["mode_limited_title"], TEXT["image_only_flow"])
            return
        if self.image_source_selector.currentIndex() == 1:
            file_paths, _ = QFileDialog.getOpenFileNames(
                self,
                TEXT["pick_batch_title"],
                str(Path("Reference/TestFiles").resolve()),
                "Images (*.png *.jpg *.jpeg *.bmp)",
            )
            if not file_paths:
                return
            supported = {".png", ".jpg", ".jpeg", ".bmp"}
            self.batch_image_paths = sorted(
                [Path(file_path) for file_path in file_paths if Path(file_path).suffix.lower() in supported]
            )
            self.batch_results = []
            self.current_result = None
            self.current_image_path = None
            self.annotated_pixmap = QPixmap()
            self.showing_annotated = False
            self.batch_showing_annotated = False
            self.current_batch_index = None
            self.detail_list.clear()
            if not self.batch_image_paths:
                self.overview_value_label.setText(TEXT["batch_empty_body"])
                QMessageBox.warning(self, TEXT["batch_empty_title"], TEXT["batch_empty_body"])
                self._update_batch_navigation()
                self._update_preview_action_visibility()
                return
            parent_dirs = {path.parent for path in self.batch_image_paths}
            self.current_batch_dir = next(iter(parent_dirs)) if len(parent_dirs) == 1 else None
            self.batch_selection_name = self.current_batch_dir.name if self.current_batch_dir is not None else "selected_images"
            self.overview_value_label.setText(TEXT["batch_loaded"].format(count=len(self.batch_image_paths)))
            self._show_batch_item(0, prefer_annotated=False)
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            TEXT["pick_image_title"],
            str(Path("Reference/TestFiles").resolve()),
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if not file_path:
            return
        self.single_image_path = Path(file_path)
        self.single_result = None
        self.single_original_pixmap = QPixmap(str(self.single_image_path))
        self.single_annotated_pixmap = QPixmap()
        self.single_showing_annotated = False
        self.current_image_path = self.single_image_path
        self.current_batch_dir = None
        self.current_batch_index = None
        self.original_pixmap = self.single_original_pixmap
        self.annotated_pixmap = QPixmap()
        self.showing_annotated = False
        self._set_preview(self.original_pixmap)
        self._update_batch_navigation()
        self.status_file_label.setText(f"{TEXT['current_file']}  {self.single_image_path.name}")
        self.overview_value_label.setText(TEXT["image_loaded"])
        self.detail_list.clear()
        self.current_result = None
        self.preview_dialog.set_pixmap(self.original_pixmap)
        self._update_preview_action_visibility()

    def select_video(self) -> None:
        if self.current_mode != "video":
            QMessageBox.information(self, TEXT["mode_limited_title"], TEXT["video_status_idle"])
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            TEXT["video_pick_title"],
            str(Path("Reference/TestFiles").resolve()),
            "Videos (*.mp4 *.avi *.mov *.mkv)",
        )
        if not file_path:
            return

        self.current_video_path = Path(file_path)
        self.current_video_result = None
        self.video_processed_frames = 0
        self.video_total_frames = None
        self.video_paused = False
        self.video_playback_finished = False
        self.video_current_frame_index = None
        self.video_next_frame_index = 0
        self.video_showing_annotated = False
        self.video_current_original_pixmap = QPixmap()
        self.video_current_annotated_pixmap = QPixmap()
        self.video_playback_timer.stop()
        self.video_original_pixmap = self._read_video_preview_pixmap(self.current_video_path)
        self.video_annotated_pixmap = QPixmap()
        self.original_pixmap = self.video_original_pixmap
        self.annotated_pixmap = self.video_annotated_pixmap
        self.showing_annotated = False
        self.status_file_label.setText(f"{TEXT['current_file']}  {self.current_video_path.name}")
        self.video_source_value.setText(self.current_video_path.name)
        self.video_status_label.setText(TEXT["video_status_idle"])
        if self.video_original_pixmap.isNull():
            self._clear_visible_display(TEXT["preview_empty"], TEXT["video_loaded"])
            self.overview_value_label.setText(TEXT["video_loaded"])
        else:
            self._set_preview(self.video_original_pixmap)
            self.preview_dialog.set_pixmap(self.video_original_pixmap)
            self.overview_value_label.setText(TEXT["video_loaded"])
            self.detail_list.clear()
        self.status_time_label.setText(f"{TEXT['elapsed']}  0.00s")
        self._update_preview_action_visibility()

    def _read_video_preview_pixmap(self, video_path: Path) -> QPixmap:
        if cv2 is None:
            return QPixmap()
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                return QPixmap()
            ok, frame = capture.read()
            if not ok:
                return QPixmap()
            return self._camera_frame_to_qpixmap(frame)
        finally:
            capture.release()

    def run_detection(self) -> None:
        if self.current_mode == "camera":
            self._toggle_camera_detection()
            return
        if self.current_mode == "video":
            self._start_video_detection()
            return
        if self.current_mode != "image":
            QMessageBox.information(self, TEXT["mode_limited_title"], TEXT["other_modes_unavailable"])
            return
        start_time = perf_counter()
        self.status_state_label.setText(f"{TEXT['state']}  {TEXT['running']}")
        try:
            if self.image_source_selector.currentIndex() == 1:
                if not self.batch_image_paths:
                    QMessageBox.warning(self, TEXT["need_batch_title"], TEXT["need_batch_body"])
                    return
                self.batch_results = [
                    self.detection_service.detect_image(
                        image_path,
                        conf=self._current_confidence(),
                        iou=self._current_iou(),
                    )
                    for image_path in self.batch_image_paths
                ]
                self._render_batch_results(self.batch_results)
            else:
                if self.current_image_path is None:
                    QMessageBox.warning(self, TEXT["need_image_title"], TEXT["need_image_body"])
                    return
                self.current_result = self.detection_service.detect_image(
                    self.single_image_path,
                    conf=self._current_confidence(),
                    iou=self._current_iou(),
                )
                self.batch_results = []
                self._render_result(self.current_result)
        except Exception as exc:
            self.status_state_label.setText(f"{TEXT['state']}  {TEXT['failed']}")
            QMessageBox.critical(self, TEXT["failed"], str(exc))
            return
        elapsed = perf_counter() - start_time
        self.status_time_label.setText(f"{TEXT['elapsed']}  {elapsed:.2f}s")
        self.status_state_label.setText(f"{TEXT['state']}  {TEXT['done']}")

    def _start_video_detection(self) -> None:
        if self.video_processing:
            QMessageBox.information(self, TEXT["video_running_block_title"], TEXT["video_running_block_body"])
            return
        if self.current_video_path is None:
            QMessageBox.warning(self, TEXT["need_video_title"], TEXT["need_video_body"])
            return

        self.video_processing = True
        self.video_processed_frames = 0
        self.video_total_frames = None
        self.video_start_time = perf_counter()
        self.status_state_label.setText(f"{TEXT['state']}  {TEXT['video_processing']}")
        self.video_status_label.setText(TEXT["video_status_running"])
        self.overview_value_label.setText(TEXT["video_overview_progress"].format(processed=0, total="?"))
        self.detail_list.clear()

        self.video_thread = QThread(self)
        self.video_worker = VideoDetectionWorker(
            self.detection_service,
            self.current_video_path,
            self._current_confidence(),
            self._current_iou(),
        )
        self.video_worker.moveToThread(self.video_thread)
        self.video_thread.started.connect(self.video_worker.run)
        self.video_worker.progress.connect(self._handle_video_progress)
        self.video_worker.frame_ready.connect(self._handle_video_frame_ready)
        self.video_worker.finished.connect(self._handle_video_finished)
        self.video_worker.failed.connect(self._handle_video_failed)
        self.video_worker.finished.connect(self.video_thread.quit)
        self.video_worker.failed.connect(self.video_thread.quit)
        self.video_thread.finished.connect(self._cleanup_video_worker)
        self.video_thread.start()
        self._update_preview_action_visibility()

    def _handle_video_progress(self, processed: int, total: object) -> None:
        self.video_processed_frames = processed
        self.video_total_frames = total if isinstance(total, int) else None
        total_text = self.video_total_frames if self.video_total_frames is not None else "?"
        self.overview_value_label.setText(
            TEXT["video_overview_progress"].format(processed=self.video_processed_frames, total=total_text)
        )

    def _handle_video_frame_ready(self, frame_result: object, total: object) -> None:
        if not isinstance(frame_result, VideoFrameResult):
            return
        self.video_processed_frames = frame_result.frame_index + 1
        self.video_total_frames = total if isinstance(total, int) else self.video_total_frames
        self.current_result = frame_result
        self.video_current_frame_index = frame_result.frame_index
        self.video_current_original_pixmap = self._coerce_video_original_pixmap(frame_result)
        self.video_current_annotated_pixmap = _to_qpixmap(frame_result.annotated_frame)
        self.video_original_pixmap = self.video_current_original_pixmap
        self.video_annotated_pixmap = self.video_current_annotated_pixmap
        self.original_pixmap = self.video_original_pixmap
        self.annotated_pixmap = self.video_annotated_pixmap
        if not self.video_paused:
            self.video_showing_annotated = not self.annotated_pixmap.isNull()
            self.showing_annotated = self.video_showing_annotated
            if self.showing_annotated:
                self._set_preview(self.annotated_pixmap)
                self.preview_dialog.set_pixmap(self.annotated_pixmap)
        self._populate_video_frame_detail(frame_result)
        self._update_preview_action_visibility()

    def _handle_video_finished(self, result: object) -> None:
        if not isinstance(result, VideoDetectionResult):
            self._handle_video_failed("Invalid video detection result.")
            return
        self.video_processing = False
        self.current_video_result = result
        elapsed = (perf_counter() - self.video_start_time) if self.video_start_time is not None else 0.0
        self._render_video_result(result, elapsed=elapsed)
        self.status_state_label.setText(f"{TEXT['state']}  {TEXT['video_done']}")
        self.video_status_label.setText(TEXT["video_status_done"])
        self._update_preview_action_visibility()

    def _handle_video_failed(self, message: str) -> None:
        self.video_processing = False
        self.status_state_label.setText(f"{TEXT['state']}  {TEXT['failed']}")
        self.video_status_label.setText(message)
        self._update_preview_action_visibility()
        QMessageBox.critical(self, TEXT["failed"], message)

    def _cleanup_video_worker(self) -> None:
        if self.video_worker is not None:
            self.video_worker.deleteLater()
        if self.video_thread is not None:
            self.video_thread.deleteLater()
        self.video_worker = None
        self.video_thread = None
        self.video_start_time = None

    def _coerce_video_original_pixmap(self, frame_result: VideoFrameResult) -> QPixmap:
        if frame_result.raw_frame is None:
            return self.video_original_pixmap
        try:
            return _to_qpixmap(frame_result.raw_frame)
        except TypeError:
            return self.video_original_pixmap

    def _show_video_frame(self, index: int) -> None:
        if self.current_video_result is None or not self.current_video_result.frames:
            return
        bounded_index = max(0, min(index, len(self.current_video_result.frames) - 1))
        frame_result = self.current_video_result.frames[bounded_index]
        self.current_result = frame_result
        self.video_current_frame_index = bounded_index
        self.video_current_original_pixmap = self._coerce_video_original_pixmap(frame_result)
        self.video_current_annotated_pixmap = _to_qpixmap(frame_result.annotated_frame)
        self.video_original_pixmap = self.video_current_original_pixmap
        self.video_annotated_pixmap = self.video_current_annotated_pixmap
        self.original_pixmap = self.video_original_pixmap
        self.annotated_pixmap = self.video_annotated_pixmap
        pixmap = self.annotated_pixmap if self.video_showing_annotated and not self.annotated_pixmap.isNull() else self.original_pixmap
        if not pixmap.isNull():
            self._set_preview(pixmap)
            self.preview_dialog.set_pixmap(pixmap)

    def _start_video_playback(self, start_index: int = 0) -> None:
        if self.current_video_result is None or not self.current_video_result.frames:
            return
        bounded_index = max(0, min(start_index, len(self.current_video_result.frames) - 1))
        self.video_showing_annotated = True
        self.showing_annotated = True
        self.video_paused = False
        self.video_playback_finished = False
        self.video_next_frame_index = bounded_index
        self.video_playback_timer.stop()
        self._play_next_video_frame()
        if not self.video_playback_finished:
            interval = int(1000 / self.current_video_result.fps) if self.current_video_result.fps else 250
            self.video_playback_timer.setInterval(max(33, interval))
            self.video_playback_timer.start()
        self._update_preview_action_visibility()

    def _play_next_video_frame(self) -> None:
        if self.current_video_result is None or self.video_paused:
            return
        if self.video_next_frame_index >= len(self.current_video_result.frames):
            self.video_playback_timer.stop()
            self.video_playback_finished = True
            self.video_paused = True
            self._update_preview_action_visibility()
            return
        self._show_video_frame(self.video_next_frame_index)
        self.video_next_frame_index += 1
        if self.video_next_frame_index >= len(self.current_video_result.frames):
            self.video_playback_timer.stop()
            self.video_playback_finished = True
            self.video_paused = True
            self._update_preview_action_visibility()

    def _toggle_video_pause(self) -> None:
        if self.video_processing:
            self.video_paused = not self.video_paused
            if not self.video_paused:
                self.video_showing_annotated = not self.annotated_pixmap.isNull()
                self.showing_annotated = self.video_showing_annotated
                if self.showing_annotated:
                    self._set_preview(self.annotated_pixmap)
                    self.preview_dialog.set_pixmap(self.annotated_pixmap)
            self._update_preview_action_visibility()
            return
        if self.current_video_result is None:
            return
        if self.video_paused:
            if self.video_playback_finished or self.video_next_frame_index >= len(self.current_video_result.frames):
                self._update_preview_action_visibility()
                return
            self.video_showing_annotated = True
            self.showing_annotated = True
            self.video_paused = False
            self.video_playback_timer.start()
        else:
            self.video_paused = True
            self.video_playback_timer.stop()
        self._update_preview_action_visibility()

    def toggle_video_preview_frame(self) -> None:
        if self.current_mode != "video" or not self.video_paused:
            return
        if self.video_showing_annotated:
            self.show_video_original_frame()
        else:
            self.show_video_annotated_frame()

    def show_video_original_frame(self) -> None:
        if self.current_mode != "video" or not self.video_paused:
            return
        self.video_showing_annotated = False
        self.showing_annotated = False
        if not self.original_pixmap.isNull():
            self._set_preview(self.original_pixmap)
            self.preview_dialog.set_pixmap(self.original_pixmap)
        self._update_preview_action_visibility()

    def show_video_annotated_frame(self) -> None:
        if self.current_mode != "video" or not self.video_paused:
            return
        self.video_showing_annotated = True
        self.showing_annotated = True
        if not self.annotated_pixmap.isNull():
            self._set_preview(self.annotated_pixmap)
            self.preview_dialog.set_pixmap(self.annotated_pixmap)
        self._update_preview_action_visibility()

    def restart_video_playback(self) -> None:
        if self.current_mode != "video" or self.current_video_result is None:
            return
        self._start_video_playback(0)

    def _toggle_camera_detection(self) -> None:
        if not self.camera_open:
            self.camera_status_label.setText(TEXT["camera_status_detect_requires_preview"])
            QMessageBox.information(self, TEXT["camera_mode"], TEXT["camera_status_detect_requires_preview"])
            return
        self.camera_detection_active = not self.camera_detection_active
        if self.camera_detection_active:
            self.camera_detection_started_at = perf_counter()
            self.camera_frame_counter = 0
            self.status_state_label.setText(f"{TEXT['state']}  {TEXT['running']}")
            self.camera_status_label.setText(TEXT["camera_status_running"])
            self.overview_value_label.setText(TEXT["camera_overview_running"])
        else:
            self.camera_current_result = None
            self.status_state_label.setText(f"{TEXT['state']}  \u6444\u50cf\u5934\u5df2\u505c\u6b62")
            self.camera_status_label.setText(TEXT["camera_status_stopped"])
            self.overview_value_label.setText(TEXT["camera_overview_stopped"])
            self.detail_list.clear()
            if self.camera_last_raw_frame is not None:
                raw_pixmap = self._camera_frame_to_qpixmap(self.camera_last_raw_frame)
                self.original_pixmap = raw_pixmap
                self.annotated_pixmap = QPixmap()
                self.showing_annotated = False
                self._set_preview(raw_pixmap)
                self.preview_dialog.set_pixmap(raw_pixmap)
        self._update_preview_action_visibility()

    def export_by_selection(self) -> None:
        if self.current_mode == "camera":
            QMessageBox.warning(self, TEXT["export_missing_title"], TEXT["export_missing_body"])
            return
        if self.current_mode == "video":
            if self.current_video_result is None:
                QMessageBox.warning(self, TEXT["export_missing_title"], TEXT["export_missing_body"])
                return
            selected = self.video_export_format_selector.currentText()
            if selected == TEXT["export_csv_only"]:
                self.export_csv()
            elif selected == TEXT["export_video_only"]:
                self.export_image()
            else:
                self.export_csv()
                self.export_image()
            return
        if self.current_result is None and not self.batch_results:
            QMessageBox.warning(self, TEXT["export_missing_title"], TEXT["export_missing_body"])
            return
        selected = self.export_format_selector.currentText()
        if selected == TEXT["export_csv_only"]:
            self.export_csv()
        elif selected == TEXT["export_image_only"]:
            self.export_image()
        else:
            self.export_csv()
            self.export_image()

    def export_csv(self) -> None:
        if self.current_mode == "video":
            if self.current_video_result is None:
                QMessageBox.warning(self, TEXT["export_missing_title"], TEXT["export_missing_body"])
                return
            output_path = self.output_dir / f"{self.current_video_result.video_path.stem}_detections.csv"
            export_video_detections_csv(self.current_video_result, output_path)
            QMessageBox.information(self, TEXT["export_done"], TEXT["export_csv_body"].format(path=output_path))
            return
        if self.image_source_selector.currentIndex() == 1:
            if not self.batch_results:
                QMessageBox.warning(self, TEXT["export_missing_title"], TEXT["export_missing_body"])
                return
            output_path = self.output_dir / f"{self._batch_display_name()}_detections.csv"
            export_detections_csv_batch(self.batch_results, output_path)
            QMessageBox.information(self, TEXT["export_done"], TEXT["export_csv_body"].format(path=output_path))
            return
        if self.current_result is None:
            QMessageBox.warning(self, TEXT["export_missing_title"], TEXT["export_missing_body"])
            return
        output_path = self.output_dir / f"{self.current_result.image_path.stem}_detections.csv"
        export_detections_csv(self.current_result, output_path)
        QMessageBox.information(self, TEXT["export_done"], TEXT["export_csv_body"].format(path=output_path))

    def export_image(self) -> None:
        if self.current_mode == "video":
            if self.current_video_result is None:
                QMessageBox.warning(self, TEXT["export_missing_title"], TEXT["export_missing_body"])
                return
            output_path = self.output_dir / f"{self.current_video_result.video_path.stem}_annotated.mp4"
            try:
                export_annotated_video(self.current_video_result, output_path)
            except Exception as exc:
                QMessageBox.critical(self, TEXT["export"], str(exc))
                return
            QMessageBox.information(self, TEXT["export_done"], TEXT["export_video_body"].format(path=output_path))
            return
        if self.image_source_selector.currentIndex() == 1:
            if not self.batch_results:
                QMessageBox.warning(self, TEXT["export_missing_title"], TEXT["export_missing_body"])
                return
            output_dir = self.output_dir / f"{self._batch_display_name()}_annotated"
            try:
                export_annotated_images(self.batch_results, output_dir)
            except Exception as exc:
                QMessageBox.critical(self, TEXT["export"], str(exc))
                return
            QMessageBox.information(self, TEXT["export_done"], TEXT["export_batch_image_body"].format(path=output_dir))
            return
        if self.current_result is None:
            QMessageBox.warning(self, TEXT["export_missing_title"], TEXT["export_missing_body"])
            return
        output_path = self.output_dir / f"{self.current_result.image_path.stem}_annotated.png"
        try:
            export_annotated_image(self.current_result, output_path)
        except Exception as exc:
            QMessageBox.critical(self, TEXT["export"], str(exc))
            return
        QMessageBox.information(self, TEXT["export_done"], TEXT["export_image_body"].format(path=output_path))

    def toggle_preview_mode(self) -> None:
        if self.current_mode == "video":
            if self.current_video_result is None or self.annotated_pixmap.isNull() or not self.video_paused:
                QMessageBox.information(self, TEXT["preview_missing_title"], TEXT["preview_missing_body"])
                return
            self.toggle_video_preview_frame()
            return
        if self.current_result is None or self.annotated_pixmap.isNull():
            QMessageBox.information(self, TEXT["preview_missing_title"], TEXT["preview_missing_body"])
            return
        self.showing_annotated = not self.showing_annotated
        if self._is_batch_source():
            self.batch_showing_annotated = self.showing_annotated
        else:
            self.single_showing_annotated = self.showing_annotated
        self._set_preview(self.annotated_pixmap if self.showing_annotated else self.original_pixmap)
        self.preview_dialog.set_pixmap(self.annotated_pixmap if self.showing_annotated else self.original_pixmap)

    def _render_result(self, result: DetectionResult) -> None:
        self.single_result = result
        self.single_image_path = result.image_path
        self.current_result = result
        self.current_image_path = result.image_path
        self.single_original_pixmap = QPixmap(str(result.image_path))
        self.single_annotated_pixmap = _to_qpixmap(result.annotated_image)
        self.original_pixmap = self.single_original_pixmap
        self.annotated_pixmap = self.single_annotated_pixmap
        self.showing_annotated = not self.annotated_pixmap.isNull()
        self.single_showing_annotated = self.showing_annotated
        self._set_preview(self.annotated_pixmap if self.showing_annotated else self.original_pixmap)
        self.preview_dialog.set_pixmap(self.annotated_pixmap if self.showing_annotated else self.original_pixmap)
        self.status_file_label.setText(f"{TEXT['current_file']}  {result.image_path.name}")
        self._populate_single_result_view(result)
        self._update_preview_action_visibility()

    def _render_batch_results(self, results: list[DetectionResult]) -> None:
        self._show_batch_item(len(results) - 1, prefer_annotated=True)
        self._populate_batch_result_view(results)

    def _render_video_result(self, result: VideoDetectionResult, elapsed: float | None = None) -> None:
        self.current_video_result = result
        self.current_video_path = result.video_path
        self.current_result = result.frames[-1] if result.frames else None
        self.video_processed_frames = result.processed_frames
        self.video_total_frames = result.total_frames
        self.video_current_frame_index = None
        self.video_next_frame_index = 0
        self.video_paused = False
        self.video_playback_finished = False
        self.status_file_label.setText(f"{TEXT['current_file']}  {result.video_path.name}")
        self.video_source_value.setText(result.video_path.name)
        if elapsed is not None:
            self.status_time_label.setText(f"{TEXT['elapsed']}  {elapsed:.2f}s")
        if result.frames:
            self._start_video_playback(0)
            self.current_result = result.frames[-1]
        else:
            self.video_annotated_pixmap = QPixmap()
            self.original_pixmap = self.video_original_pixmap
            self.annotated_pixmap = self.video_annotated_pixmap
        self._populate_video_result_view(result)
        self._update_preview_action_visibility()

    def _populate_video_frame_detail(self, frame_result: VideoFrameResult) -> None:
        total_text = self.video_total_frames if self.video_total_frames is not None else "?"
        self.overview_value_label.setText(
            TEXT["video_overview_progress"].format(processed=self.video_processed_frames, total=total_text)
        )
        self.detail_list.clear()
        for detection in frame_result.detections:
            bbox_text = ", ".join(f"{value:.1f}" for value in detection.bbox)
            self.detail_list.addItem(
                TEXT["video_detail_line"].format(
                    frame_index=frame_result.frame_index,
                    label=detection.label,
                    confidence=detection.confidence,
                    bbox=bbox_text,
                )
            )

    def _populate_video_result_view(self, result: VideoDetectionResult) -> None:
        counts = "\u3001".join(f"{label} {count}" for label, count in sorted(result.counts_by_label.items()))
        total_text = result.total_frames if result.total_frames is not None else "?"
        self.overview_value_label.setText(
            TEXT["video_overview_frames"].format(processed=result.processed_frames, total=total_text)
            + "\n"
            + TEXT["video_overview_total"].format(count=result.total_detections)
            + "\n"
            + (counts if counts else TEXT["overview_none"])
        )
        self.detail_list.clear()
        for frame in result.frames:
            for detection in frame.detections:
                bbox_text = ", ".join(f"{value:.1f}" for value in detection.bbox)
                self.detail_list.addItem(
                    TEXT["video_detail_line"].format(
                        frame_index=frame.frame_index,
                        label=detection.label,
                        confidence=detection.confidence,
                        bbox=bbox_text,
                    )
                )

    def _populate_single_result_view(self, result: DetectionResult) -> None:
        counts = "\u3001".join(f"{label} {count}" for label, count in sorted(result.counts_by_label.items()))
        self.overview_value_label.setText(
            TEXT["overview_total"].format(count=result.total_detections)
            + "\n"
            + (counts if counts else TEXT["overview_none"])
        )
        self.detail_list.clear()
        for detection in result.detections:
            bbox_text = ", ".join(f"{value:.1f}" for value in detection.bbox)
            self.detail_list.addItem(
                TEXT["detail_line"].format(
                    label=detection.label,
                    confidence=detection.confidence,
                    bbox=bbox_text,
                )
            )

    def _populate_batch_result_view(self, results: list[DetectionResult]) -> None:
        total_images = len(results)
        total_detections = sum(result.total_detections for result in results)
        combined_counts: dict[str, int] = {}
        for result in results:
            for label, count in result.counts_by_label.items():
                combined_counts[label] = combined_counts.get(label, 0) + count
        counts = "\u3001".join(f"{label} {count}" for label, count in sorted(combined_counts.items()))
        self.overview_value_label.setText(
            TEXT["overview_batch"].format(count=total_images)
            + "\n"
            + TEXT["overview_total"].format(count=total_detections)
            + "\n"
            + (counts if counts else TEXT["overview_none"])
        )
        self.detail_list.clear()
        for result in results:
            for detection in result.detections:
                bbox_text = ", ".join(f"{value:.1f}" for value in detection.bbox)
                self.detail_list.addItem(
                    TEXT["detail_batch_line"].format(
                        file_name=result.image_path.name,
                        label=detection.label,
                        confidence=detection.confidence,
                        bbox=bbox_text,
                    )
                )

    def _clear_preview(self, text: str = TEXT["preview_empty"]) -> None:
        self.current_preview_pixmap = QPixmap()
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText(text)
        self.preview_dialog.set_pixmap(QPixmap())

    def _clear_visible_display(
        self,
        preview_text: str = TEXT["preview_empty"],
        overview_text: str | None = None,
    ) -> None:
        self._clear_preview(preview_text)
        if overview_text is not None:
            self.overview_value_label.setText(overview_text)
        self.detail_list.clear()

    def _is_batch_source(self) -> bool:
        return hasattr(self, "image_source_selector") and self.image_source_selector.currentIndex() == 1

    def _remember_current_single_state(self, force: bool = False) -> None:
        if self.current_mode != "image" or self.current_image_path is None:
            return
        if not force and self._is_batch_source():
            return
        self.single_image_path = self.current_image_path
        self.single_result = self.current_result
        self.single_original_pixmap = self.original_pixmap
        self.single_annotated_pixmap = self.annotated_pixmap
        self.single_showing_annotated = self.showing_annotated

    def _restore_source_state(self) -> None:
        if self.current_mode != "image":
            return
        if self._is_batch_source():
            self._restore_batch_state()
        else:
            self._restore_single_state()
        self._update_batch_navigation()
        self._update_preview_action_visibility()

    def _restore_single_state(self) -> None:
        self.current_batch_index = None
        self.current_image_path = self.single_image_path
        self.current_result = self.single_result
        self.original_pixmap = self.single_original_pixmap
        self.annotated_pixmap = self.single_annotated_pixmap
        self.showing_annotated = self.single_showing_annotated and not self.annotated_pixmap.isNull()
        if self.single_image_path is None:
            self.status_file_label.setText(f"{TEXT['current_file']}  \u672a\u9009\u62e9")
            self._clear_visible_display(TEXT["preview_empty"], TEXT["no_result"])
            return
        self.status_file_label.setText(f"{TEXT['current_file']}  {self.single_image_path.name}")
        if self.single_result is None:
            self._set_preview(self.original_pixmap)
            self.preview_dialog.set_pixmap(self.original_pixmap)
            self.overview_value_label.setText(TEXT["image_loaded"])
            self.detail_list.clear()
            return
        pixmap = self.annotated_pixmap if self.showing_annotated else self.original_pixmap
        self._set_preview(pixmap)
        self.preview_dialog.set_pixmap(pixmap)
        self._populate_single_result_view(self.single_result)

    def _restore_video_state(self) -> None:
        if self.current_video_path is None:
            self.status_file_label.setText(f"{TEXT['current_file']}  \u672a\u9009\u62e9")
            self.video_source_value.setText(TEXT["need_video_body"])
            self.video_status_label.setText(TEXT["video_status_idle"])
            overview_text = TEXT["need_video_body"] if self.single_image_path is not None else TEXT["video_overview_ready"]
            self._clear_visible_display(TEXT["preview_empty"], overview_text)
            return
        self.current_result = self.current_video_result.frames[-1] if self.current_video_result and self.current_video_result.frames else None
        self.current_image_path = None
        self.original_pixmap = self.video_original_pixmap
        self.annotated_pixmap = self.video_annotated_pixmap
        self.showing_annotated = self.video_showing_annotated and not self.annotated_pixmap.isNull()
        self.status_file_label.setText(f"{TEXT['current_file']}  {self.current_video_path.name}")
        self.video_source_value.setText(self.current_video_path.name)
        if self.current_video_result is None:
            self.video_status_label.setText(TEXT["video_status_idle"] if not self.video_processing else TEXT["video_status_running"])
            if self.video_original_pixmap.isNull():
                self._clear_visible_display(TEXT["preview_empty"], TEXT["video_loaded"])
            else:
                self._set_preview(self.video_original_pixmap)
                self.preview_dialog.set_pixmap(self.video_original_pixmap)
                self.overview_value_label.setText(TEXT["video_loaded"])
                self.detail_list.clear()
            return
        if self.video_current_frame_index is not None:
            self._show_video_frame(self.video_current_frame_index)
            self.video_status_label.setText(TEXT["video_status_done"])
            return
        pixmap = self.annotated_pixmap if self.showing_annotated else self.original_pixmap
        if not pixmap.isNull():
            self._set_preview(pixmap)
            self.preview_dialog.set_pixmap(pixmap)
        self.video_status_label.setText(TEXT["video_status_done"])
        self._populate_video_result_view(self.current_video_result)

    def _restore_batch_state(self) -> None:
        if not self.batch_image_paths:
            self.current_batch_index = None
            self.current_image_path = None
            self.current_result = None
            self.status_file_label.setText(f"{TEXT['current_file']}  \u672a\u9009\u62e9")
            self._clear_visible_display(TEXT["preview_empty"], TEXT["need_batch_body"])
            return
        self._show_batch_item(self.current_batch_index if self.current_batch_index is not None else 0)

    def _update_preview_action_visibility(self) -> None:
        has_selection = bool(self.batch_image_paths) if self._is_batch_source() else self.single_image_path is not None
        is_batch = self.current_mode == "image" and self._is_batch_source() and bool(self.batch_image_paths)
        if self.current_mode == "camera":
            self.open_action_button.setText(TEXT["close_camera"] if self.camera_open else TEXT["open_camera"])
            self.detect_action_button.setText(TEXT["stop_detect"] if self.camera_detection_active else TEXT["detect"])
            self.preview_action_button.setVisible(True)
            self.video_original_button.setVisible(False)
            self.video_result_button.setVisible(False)
            self.video_restart_button.setVisible(False)
            self.export_action_button.setVisible(False)
            self.close_current_button.setVisible(False)
            self.close_all_button.setVisible(False)
            if self.preview_hint_label is not None:
                self.preview_hint_label.setText(TEXT["preview_hint"])
            return
        if self.current_mode == "video":
            self.open_action_button.setText(TEXT["open"])
            self.detect_action_button.setText(TEXT["detect"])
            self.open_action_button.setEnabled(not self.video_processing)
            self.detect_action_button.setEnabled(self.current_video_path is not None and not self.video_processing)
            self.preview_action_button.setVisible(True)
            self.preview_action_button.setEnabled(
                not self.video_original_pixmap.isNull() or not self.video_annotated_pixmap.isNull()
            )
            video_ready = self.current_video_result is not None and not self.video_processing
            can_toggle_frame_view = self.video_paused and (
                not self.original_pixmap.isNull() or not self.annotated_pixmap.isNull()
            )
            self.video_original_button.setVisible(True)
            self.video_original_button.setText(
                TEXT["video_original"] if self.video_showing_annotated else TEXT["video_result"]
            )
            self.video_result_button.setVisible(False)
            self.video_restart_button.setVisible(True)
            self.video_original_button.setEnabled(can_toggle_frame_view)
            self.video_restart_button.setEnabled(video_ready)
            self.export_action_button.setVisible(True)
            self.export_action_button.setEnabled(self.current_video_result is not None and not self.video_processing)
            self.close_current_button.setVisible(self.current_video_path is not None and not self.video_processing)
            self.close_all_button.setVisible(False)
            if self.preview_hint_label is not None:
                self.preview_hint_label.setText(TEXT["video_preview_hint"])
            return
        self.video_original_button.setVisible(False)
        self.video_result_button.setVisible(False)
        self.video_restart_button.setVisible(False)
        if self.preview_hint_label is not None:
            self.preview_hint_label.setText(TEXT["preview_hint"])
        self.open_action_button.setText(TEXT["open"])
        self.detect_action_button.setText(TEXT["detect"])
        self.open_action_button.setEnabled(True)
        self.detect_action_button.setEnabled(True)
        self.preview_action_button.setVisible(True)
        self.preview_action_button.setEnabled(True)
        self.export_action_button.setVisible(True)
        self.export_action_button.setEnabled(True)
        self.close_current_button.setVisible(self.current_mode == "image" and has_selection)
        self.close_all_button.setVisible(is_batch)

    def close_current_display(self) -> None:
        if self.current_mode == "video":
            if self.video_processing or self.current_video_path is None:
                return
            self._clear_video_state()
            return
        if self.current_mode != "image":
            return
        if self._is_batch_source():
            if self.current_batch_index is None:
                return
            remove_index = self.current_batch_index
            self.batch_image_paths.pop(remove_index)
            if remove_index < len(self.batch_results):
                self.batch_results.pop(remove_index)
            if not self.batch_image_paths:
                self._clear_batch_state()
                return
            next_index = min(remove_index, len(self.batch_image_paths) - 1)
            self._show_batch_item(next_index, prefer_annotated=self.batch_showing_annotated)
        else:
            self._remember_current_single_state()
            if self.single_image_path is None:
                return
            self._clear_single_state()

    def close_all_displays(self) -> None:
        if self.current_mode != "image" or not self._is_batch_source() or not self.batch_image_paths:
            return
        self._clear_batch_state()
        self._update_preview_action_visibility()

    def _clear_single_state(self) -> None:
        self.single_image_path = None
        self.single_result = None
        self.single_original_pixmap = QPixmap()
        self.single_annotated_pixmap = QPixmap()
        self.single_showing_annotated = False
        self.current_batch_index = None
        self.current_image_path = None
        self.current_result = None
        self.original_pixmap = QPixmap()
        self.annotated_pixmap = QPixmap()
        self.showing_annotated = False
        self.status_file_label.setText(f"{TEXT['current_file']}  \u672a\u9009\u62e9")
        self._clear_visible_display(TEXT["preview_empty"], TEXT["no_result"])
        self._update_batch_navigation()
        self._update_preview_action_visibility()

    def _clear_video_state(self) -> None:
        self.video_playback_timer.stop()
        self.current_video_path = None
        self.current_video_result = None
        self.video_original_pixmap = QPixmap()
        self.video_annotated_pixmap = QPixmap()
        self.video_showing_annotated = False
        self.video_processed_frames = 0
        self.video_total_frames = None
        self.video_paused = False
        self.video_playback_finished = False
        self.video_current_frame_index = None
        self.video_next_frame_index = 0
        self.video_current_original_pixmap = QPixmap()
        self.video_current_annotated_pixmap = QPixmap()
        self.current_result = None
        self.original_pixmap = QPixmap()
        self.annotated_pixmap = QPixmap()
        self.showing_annotated = False
        self.status_file_label.setText(f"{TEXT['current_file']}  \u672a\u9009\u62e9")
        self.video_source_value.setText(TEXT["need_video_body"])
        self.video_status_label.setText(TEXT["video_status_idle"])
        self.status_time_label.setText(f"{TEXT['elapsed']}  0.00s")
        self._clear_visible_display(TEXT["preview_empty"], TEXT["need_video_body"])
        self._update_preview_action_visibility()

    def _clear_batch_state(self) -> None:
        self.current_batch_dir = None
        self.current_batch_index = None
        self.batch_selection_name = "selected_images"
        self.batch_image_paths = []
        self.batch_results = []
        self.batch_showing_annotated = False
        self.current_image_path = None
        self.current_result = None
        self.original_pixmap = QPixmap()
        self.annotated_pixmap = QPixmap()
        self.showing_annotated = False
        self.status_file_label.setText(f"{TEXT['current_file']}  \u672a\u9009\u62e9")
        self._clear_visible_display(TEXT["preview_empty"], TEXT["need_batch_body"])
        self._update_batch_navigation()
        self._update_preview_action_visibility()

    def _set_preview(self, pixmap: QPixmap) -> None:
        self.current_preview_pixmap = pixmap
        self.preview_zoom_factor = 1.0
        self._update_main_preview_display()

    def _update_main_preview_display(self) -> None:
        pixmap = self.current_preview_pixmap
        if pixmap.isNull():
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText(TEXT["preview_empty"])
            stage_min_length = _preview_stage_min_length()
            self.image_label.resize(stage_min_length, stage_min_length)
            return
        self.image_label.setText("")
        viewport = self.preview_scroll_area.viewport().size()
        if viewport.width() <= 0 or viewport.height() <= 0:
            viewport = self.preview_scroll_area.size()
        if pixmap.width() > 0 and pixmap.height() > 0:
            self.preview_base_scale = min(
                viewport.width() / pixmap.width(),
                viewport.height() / pixmap.height(),
                1.0,
            )
        else:
            self.preview_base_scale = 1.0
        scale = max(0.1, self.preview_base_scale * self.preview_zoom_factor)
        scaled = pixmap.scaled(
            max(1, int(pixmap.width() * scale)),
            max(1, int(pixmap.height() * scale)),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.image_label.resize(scaled.size())

    def closeEvent(self, event) -> None:
        self._stop_camera_preview(clear_display=False)
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self.current_preview_pixmap.isNull():
            self._update_main_preview_display()


def create_app(model_path: str | Path, output_dir: str | Path) -> QApplication:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(DetectionService(model_path), output_dir)
    window.show()
    app._main_window = window  # type: ignore[attr-defined]
    return app
