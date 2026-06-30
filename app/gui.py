from __future__ import annotations

import csv
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.export import export_annotated_image, export_detections_csv
from app.infer import DetectionResult, DetectionService


def _to_qpixmap(image_data: object) -> QPixmap:
    if image_data is None:
        return QPixmap()
    if isinstance(image_data, bytes):
        qimage = QImage.fromData(image_data)
        return QPixmap.fromImage(qimage)
    if hasattr(image_data, "shape"):
        height, width, channels = image_data.shape
        bytes_per_line = channels * width
        qimage = QImage(image_data.data, width, height, bytes_per_line, QImage.Format_BGR888)
        return QPixmap.fromImage(qimage.copy())
    raise TypeError(f"Unsupported image data type: {type(image_data)!r}")


class MainWindow(QMainWindow):
    def __init__(self, detection_service: DetectionService, output_dir: str | Path):
        super().__init__()
        self.detection_service = detection_service
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_result: DetectionResult | None = None
        self.current_image_path: Path | None = None

        self.setWindowTitle("Litchi Detector")
        self.resize(1200, 720)

        self.image_label = QLabel("请选择一张图片开始检测")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(720, 540)
        self.image_label.setStyleSheet("border: 1px solid #cccccc; background: #fafafa;")

        self.summary_label = QLabel("当前没有检测结果")
        self.details_list = QListWidget()

        select_button = QPushButton("选择图片")
        select_button.clicked.connect(self.select_image)

        detect_button = QPushButton("执行检测")
        detect_button.clicked.connect(self.run_detection)

        export_button = QPushButton("导出 CSV")
        export_button.clicked.connect(self.export_csv)

        export_image_button = QPushButton("导出结果图")
        export_image_button.clicked.connect(self.export_image)

        button_layout = QHBoxLayout()
        button_layout.addWidget(select_button)
        button_layout.addWidget(detect_button)
        button_layout.addWidget(export_button)
        button_layout.addWidget(export_image_button)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.summary_label)
        right_layout.addWidget(self.details_list)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.image_label, stretch=3)
        main_layout.addLayout(right_layout, stretch=2)

        page_layout = QVBoxLayout()
        page_layout.addLayout(button_layout)
        page_layout.addLayout(main_layout)

        container = QWidget()
        container.setLayout(page_layout)
        self.setCentralWidget(container)

    def select_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择待检测图片",
            str(Path("Reference/TestFiles").resolve()),
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if not file_path:
            return
        self.current_image_path = Path(file_path)
        pixmap = QPixmap(str(self.current_image_path))
        self._set_preview(pixmap)
        self.summary_label.setText(f"已选择: {self.current_image_path.name}")
        self.details_list.clear()
        self.current_result = None

    def run_detection(self) -> None:
        if self.current_image_path is None:
            QMessageBox.warning(self, "未选择图片", "请先选择一张图片。")
            return

        try:
            self.current_result = self.detection_service.detect_image(self.current_image_path)
        except Exception as exc:
            QMessageBox.critical(self, "检测失败", str(exc))
            return

        self._render_result(self.current_result)

    def export_csv(self) -> None:
        if self.current_result is None:
            QMessageBox.warning(self, "无结果可导出", "请先执行一次检测。")
            return

        output_path = self.output_dir / f"{self.current_result.image_path.stem}_detections.csv"
        export_detections_csv(self.current_result, output_path)
        QMessageBox.information(self, "导出完成", f"CSV 已保存到:\n{output_path}")

    def export_image(self) -> None:
        if self.current_result is None:
            QMessageBox.warning(self, "无结果可导出", "请先执行一次检测。")
            return

        output_path = self.output_dir / f"{self.current_result.image_path.stem}_annotated.png"
        try:
            export_annotated_image(self.current_result, output_path)
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        QMessageBox.information(self, "导出完成", f"结果图已保存到:\n{output_path}")

    def _render_result(self, result: DetectionResult) -> None:
        self._set_preview(_to_qpixmap(result.annotated_image))
        counts_text = ", ".join(f"{label}: {count}" for label, count in sorted(result.counts_by_label.items()))
        self.summary_label.setText(
            f"总检测数: {result.total_detections}\n{counts_text if counts_text else '未检测到目标'}"
        )
        self.details_list.clear()
        for detection in result.detections:
            self.details_list.addItem(
                f"{detection.label} | conf={detection.confidence:.2f} | bbox={tuple(round(v, 1) for v in detection.bbox)}"
            )

    def _set_preview(self, pixmap: QPixmap) -> None:
        if pixmap.isNull():
            self.image_label.setText("无法显示当前图片")
            return
        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)


def create_app(model_path: str | Path, output_dir: str | Path) -> QApplication:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(DetectionService(model_path), output_dir)
    window.show()
    app._main_window = window  # type: ignore[attr-defined]
    return app
