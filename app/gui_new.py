from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from app.gui import MainWindow, TEXT
from app.infer import DetectionService


class MainWindowNew(MainWindow):
    """Alternative visual shell that reuses the existing GUI behavior."""

    def _build_ui(self) -> None:
        self.setWindowTitle("荔枝检测与计数智能系统 · New GUI")
        self.resize(1680, 1000)

        page = QWidget()
        page.setObjectName("new-root-shell")
        page_layout = QVBoxLayout()
        page_layout.setContentsMargins(22, 20, 22, 20)
        page_layout.setSpacing(18)

        page_layout.addWidget(self._build_new_top_bar())
        page_layout.addWidget(self._build_new_body(), 1)

        page.setLayout(page_layout)
        self.setCentralWidget(page)

    def _build_new_top_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("new-top-bar")
        layout = QHBoxLayout()
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(18)

        title_card = QFrame()
        title_card.setObjectName("new-title-card")
        title_layout = QVBoxLayout()
        title_layout.setContentsMargins(18, 10, 18, 10)
        title_layout.setSpacing(4)
        title = QLabel("复杂果园荔枝智能检测台")
        title.setObjectName("new-app-title")
        subtitle = QLabel("图像 · 视频 · 摄像头一体化检测与计数")
        subtitle.setObjectName("new-app-subtitle")
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        title_card.setLayout(title_layout)

        right_stack = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)
        right_layout.addWidget(self._build_mode_header())
        right_layout.addWidget(self._build_status_row())
        right_stack.setLayout(right_layout)

        layout.addWidget(title_card, 2)
        layout.addWidget(right_stack, 5)
        bar.setLayout(layout)
        return bar

    def _build_new_body(self) -> QWidget:
        body = QWidget()
        body.setObjectName("new-workbench")
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)

        layout.addWidget(self._build_new_side_panel(), 2)
        layout.addWidget(self._build_new_preview_panel(), 5)
        layout.addWidget(self._build_new_result_panel(), 3)

        body.setLayout(layout)
        return body

    def _build_new_side_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("new-side-panel")
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        layout.addWidget(self._build_config_panel())
        layout.addStretch(1)
        panel.setLayout(layout)
        return panel

    def _build_new_preview_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("new-preview-panel")
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_preview_panel())
        panel.setLayout(layout)
        return panel

    def _build_new_result_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("new-result-panel")
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_result_panel())
        panel.setLayout(layout)
        return panel

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background: #d6dcc7;
            }
            QWidget {
                color: #203228;
                font-size: 19px;
            }
            #new-root-shell {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #f2efe3,
                    stop:0.42 #dbe6d0,
                    stop:1 #bfd1bc
                );
            }
            QFrame#new-top-bar,
            QFrame#new-side-panel,
            QFrame#new-preview-panel,
            QFrame#new-result-panel {
                background: rgba(255, 252, 244, 0.72);
                border: 1px solid rgba(80, 104, 76, 0.28);
                border-radius: 24px;
            }
            QFrame#new-title-card {
                background: #244533;
                border: 1px solid #183226;
                border-radius: 20px;
            }
            QLabel#new-app-title {
                color: #fff8e8;
                font-size: 30px;
                font-weight: 800;
                letter-spacing: 1px;
            }
            QLabel#new-app-subtitle {
                color: #dfe9c6;
                font-size: 16px;
            }
            QWidget#mode-switcher {
                background: transparent;
            }
            QFrame#status-info-bar,
            QFrame#left-config-panel,
            QFrame#left-result-panel,
            QFrame#preview-panel {
                background: #fffaf0;
                border: 1px solid rgba(68, 91, 59, 0.24);
                border-radius: 20px;
            }
            QFrame#status-metric-card,
            QFrame#result-overview-card,
            QFrame#result-detail-card {
                background: #f7f1df;
                border: 1px solid rgba(66, 83, 53, 0.2);
                border-radius: 16px;
            }
            QLabel#section-title {
                color: #1f3b2b;
                font-size: 26px;
                font-weight: 800;
            }
            QLabel#preview-hint-label {
                color: #63705e;
                font-size: 15px;
            }
            QPushButton {
                background: #244533;
                border: 1px solid #173124;
                border-radius: 16px;
                color: #fff8e8;
                font-size: 21px;
                padding: 12px 22px;
                min-width: 122px;
            }
            QPushButton:hover {
                background: #315d43;
            }
            QPushButton:pressed {
                background: #183226;
            }
            QPushButton:checked {
                background: #c56b3f;
                border-color: #9e4d27;
                color: #fff7ec;
            }
            QPushButton:disabled {
                background: #a7b1a0;
                border-color: #8f9a87;
                color: #eef1e8;
            }
            QPushButton#preview-action-close-current {
                min-width: 56px;
                max-width: 56px;
                padding: 12px 0;
                font-size: 24px;
            }
            QComboBox,
            QListWidget {
                background: #fffdf7;
                border: 1px solid rgba(68, 91, 59, 0.28);
                border-radius: 14px;
                padding: 8px 12px;
                font-size: 19px;
            }
            QScrollArea#preview-scroll-area {
                background: #eef2e5;
                border: 1px dashed rgba(58, 78, 55, 0.35);
                border-radius: 18px;
            }
            QSlider::groove:horizontal {
                background: #d7dfcc;
                height: 10px;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: #c56b3f;
                width: 22px;
                margin: -6px 0;
                border-radius: 11px;
            }
            QCheckBox {
                spacing: 12px;
                font-size: 19px;
            }
            QCheckBox::indicator {
                width: 24px;
                height: 24px;
            }
            """
        )


def create_new_app(model_path: str | Path, output_dir: str | Path) -> QApplication:
    app = QApplication.instance() or QApplication([])
    window = MainWindowNew(DetectionService(model_path), output_dir)
    window.show()
    app._main_window = window  # type: ignore[attr-defined]
    return app
