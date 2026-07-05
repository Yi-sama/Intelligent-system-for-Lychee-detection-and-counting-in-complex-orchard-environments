import csv
import shutil
import unittest
from base64 import b64decode
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - depends on local env
    np = None
try:
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QFrame,
        QLabel,
        QListWidget,
        QPushButton,
        QSlider,
        QStackedWidget,
        QWidget,
    )
except ModuleNotFoundError:  # pragma: no cover - depends on local env
    QApplication = None
    QCheckBox = None
    QComboBox = None
    QDialog = None
    QFrame = None
    QLabel = None
    QListWidget = None
    QPixmap = None
    QPushButton = None
    QSlider = None
    QStackedWidget = None
    QWidget = None

from app.export import (
    export_annotated_image,
    export_detections_csv,
)
try:
    from app.gui import MainWindow, TEXT
except ModuleNotFoundError:  # pragma: no cover - depends on local env
    MainWindow = None
    TEXT = {}
try:
    from app.gui_new import MainWindowNew
except ModuleNotFoundError:  # pragma: no cover - depends on local env
    MainWindowNew = None
from app.infer import (
    CameraFrameResult,
    Detection,
    DetectionResult,
    DetectionService,
    VideoDetectionResult,
    VideoFrameResult,
)


VALID_PNG_BYTES = b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aAt8AAAAASUVORK5CYII="
)


class FakeBoxes:
    def __init__(self):
        self.cls = [0, 1, 0]
        self.conf = [0.95, 0.8, 0.67]
        self.xyxy = [
            [10, 20, 30, 40],
            [15, 25, 35, 45],
            [50, 60, 70, 80],
        ]


class FakeResult:
    def __init__(self):
        self.names = {0: "lychee", 1: "branch"}
        self.boxes = FakeBoxes()

    def plot(self):
        return b"annotated-image"


class FakeModel:
    def __init__(self):
        self.predict_calls = []

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        return [FakeResult()]


def fake_model_loader(_model_path):
    return FakeModel()


class FakeVideoResult:
    def __init__(self, names, classes, confidences, boxes, annotated_frame):
        self.names = names
        self.boxes = type(
            "Boxes",
            (),
            {
                "cls": classes,
                "conf": confidences,
                "xyxy": boxes,
            },
        )()
        self._annotated_frame = annotated_frame

    def plot(self):
        return self._annotated_frame


class FakeVideoModel:
    def __init__(self, results):
        self.results = results
        self.predict_calls = []

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        return list(self.results)


class FakeVideoCaptureForInfer:
    def __init__(self, frame_count=0, fps=0.0, opened=True):
        self.frame_count = frame_count
        self.fps = fps
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, prop_id):
        if prop_id == 7:
            return self.frame_count
        if prop_id == 5:
            return self.fps
        return 0

    def release(self):
        self.released = True


class FakeCv2InferModule:
    CAP_PROP_FRAME_COUNT = 7
    CAP_PROP_FPS = 5

    def __init__(self, capture):
        self.capture = capture
        self.opened_sources = []

    def VideoCapture(self, source):
        self.opened_sources.append(source)
        return self.capture


class FakeVideoWriter:
    def __init__(self, output_path, fourcc, fps, frame_size):
        self.output_path = output_path
        self.fourcc = fourcc
        self.fps = fps
        self.frame_size = frame_size
        self.frames = []
        self.released = False

    def write(self, frame):
        self.frames.append(frame)

    def release(self):
        self.released = True


class FakeCv2ExportModule:
    def __init__(self):
        self.writer_calls = []
        self.writers = []
        self.fourcc_calls = []

    def VideoWriter_fourcc(self, *codec):
        self.fourcc_calls.append(codec)
        return "fourcc"

    def VideoWriter(self, output_path, fourcc, fps, frame_size):
        self.writer_calls.append((output_path, fourcc, fps, frame_size))
        writer = FakeVideoWriter(output_path, fourcc, fps, frame_size)
        self.writers.append(writer)
        return writer


class FakeCameraCapture:
    def __init__(self, frame=None, opened=True):
        self.frame = frame
        self.opened = opened
        self.released = False
        self.set_calls = []

    def isOpened(self):
        return self.opened

    def read(self):
        if self.frame is None:
            return False, None
        return True, self.frame

    def set(self, prop_id, value):
        self.set_calls.append((prop_id, value))
        return True

    def release(self):
        self.released = True


class SequencedCameraCapture:
    def __init__(self, reads, opened=True):
        self.reads = list(reads)
        self.opened = opened
        self.released = False
        self.set_calls = []

    def isOpened(self):
        return self.opened

    def read(self):
        if self.reads:
            return self.reads.pop(0)
        return False, None

    def set(self, prop_id, value):
        self.set_calls.append((prop_id, value))
        return True

    def release(self):
        self.released = True


class FakeCv2Module:
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    COLOR_BGR2RGB = object()
    FONT_HERSHEY_SIMPLEX = object()
    LINE_AA = object()

    def __init__(self, capture):
        self._capture = capture

    def VideoCapture(self, _index):
        return self._capture

    def cvtColor(self, frame, _conversion):
        return frame[..., ::-1]

    def rectangle(self, image, _pt1, _pt2, _color, _thickness):
        return image

    def getTextSize(self, text, _font, _scale, _thickness):
        return ((max(1, len(text)) * 8, 12), 4)

    def putText(self, image, _text, _org, _font, _scale, _color, _thickness, _line_type):
        return image

    def imwrite(self, output_path, _frame):
        Path(output_path).write_bytes(b"fake-camera-frame")
        return True


class SequencedCameraCv2Module(FakeCv2Module):
    CAP_DSHOW = 700
    CAP_MSMF = 1400

    def __init__(self, captures):
        super().__init__(captures[0] if captures else None)
        self.captures = list(captures)
        self.video_capture_calls = []

    def VideoCapture(self, index, backend=None):
        self.video_capture_calls.append((index, backend))
        if self.captures:
            return self.captures.pop(0)
        return SequencedCameraCapture([], opened=False)


class DetectionServiceTests(unittest.TestCase):
    def test_detect_image_returns_summary_and_detections(self):
        temp_path = Path("tests/.tmp/detect_case")
        if temp_path.exists():
            shutil.rmtree(temp_path)
        temp_path.mkdir(parents=True, exist_ok=True)
        try:
            image_path = temp_path / "sample.jpg"
            image_path.write_bytes(b"fake-image")

            service = DetectionService("fake.pt", model_loader=fake_model_loader)
            result = service.detect_image(image_path)

            self.assertEqual(result.image_path, image_path)
            self.assertEqual(result.total_detections, 3)
            self.assertEqual(result.counts_by_label, {"lychee": 2, "branch": 1})
            self.assertEqual(result.annotated_image, b"annotated-image")
            self.assertEqual(result.detections[0].label, "lychee")
            self.assertEqual(result.detections[1].confidence, 0.8)
            self.assertEqual(result.detections[2].bbox, (50.0, 60.0, 70.0, 80.0))
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)

    def test_detect_image_passes_conf_and_iou_to_model_predict(self):
        temp_path = Path("tests/.tmp/detect_threshold_case")
        if temp_path.exists():
            shutil.rmtree(temp_path)
        temp_path.mkdir(parents=True, exist_ok=True)
        try:
            image_path = temp_path / "sample.jpg"
            image_path.write_bytes(b"fake-image")
            service = DetectionService("fake.pt", model_loader=fake_model_loader)

            service.detect_image(image_path, conf=0.33, iou=0.61)

            self.assertEqual(
                service._model.predict_calls[-1],
                {
                    "source": str(image_path),
                    "verbose": False,
                    "conf": 0.33,
                    "iou": 0.61,
                },
            )
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)

    def test_detect_video_aggregates_frame_results_and_reports_progress(self):
        from app.infer import VideoDetectionResult, VideoFrameResult

        temp_path = Path("tests/.tmp/detect_video_case")
        if temp_path.exists():
            shutil.rmtree(temp_path)
        temp_path.mkdir(parents=True, exist_ok=True)
        try:
            video_path = temp_path / "sample.mp4"
            video_path.write_bytes(b"fake-video")
            model = FakeVideoModel(
                [
                    FakeVideoResult(
                        names={0: "lychee", 1: "branch"},
                        classes=[0, 1],
                        confidences=[0.91, 0.52],
                        boxes=[[1, 2, 11, 12], [3, 4, 13, 14]],
                        annotated_frame="frame-0",
                    ),
                    FakeVideoResult(
                        names={0: "lychee", 1: "branch"},
                        classes=[0],
                        confidences=[0.88],
                        boxes=[[5, 6, 15, 16]],
                        annotated_frame="frame-1",
                    ),
                ]
            )
            fake_cv2 = FakeCv2InferModule(FakeVideoCaptureForInfer(frame_count=2, fps=24.0))
            progress_updates = []
            service = DetectionService("fake.pt", model_loader=lambda _: model)

            with patch("app.infer.cv2", fake_cv2):
                result = service.detect_video(
                    video_path,
                    conf=0.25,
                    iou=0.45,
                    progress_callback=lambda processed, total: progress_updates.append((processed, total)),
                )

            self.assertIsInstance(result, VideoDetectionResult)
            self.assertEqual(result.video_path, video_path)
            self.assertEqual(result.total_frames, 2)
            self.assertEqual(result.processed_frames, 2)
            self.assertEqual(result.fps, 24.0)
            self.assertEqual(result.total_detections, 3)
            self.assertEqual(result.counts_by_label, {"lychee": 2, "branch": 1})
            self.assertEqual(progress_updates, [(1, 2), (2, 2)])
            self.assertEqual(
                model.predict_calls[-1],
                {
                    "source": str(video_path),
                    "verbose": False,
                    "stream": True,
                    "conf": 0.25,
                    "iou": 0.45,
                },
            )
            self.assertEqual(len(result.frames), 2)
            self.assertIsInstance(result.frames[0], VideoFrameResult)
            self.assertEqual(result.frames[0].frame_index, 0)
            self.assertEqual(result.frames[0].timestamp_seconds, 0.0)
            self.assertEqual(result.frames[0].annotated_frame, "frame-0")
            self.assertEqual(result.frames[1].frame_index, 1)
            self.assertAlmostEqual(result.frames[1].timestamp_seconds, 1 / 24)
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)

    def test_detect_video_emits_frame_callback_for_each_processed_frame(self):
        from app.infer import VideoFrameResult

        temp_path = Path("tests/.tmp/detect_video_frame_callback_case")
        if temp_path.exists():
            shutil.rmtree(temp_path)
        temp_path.mkdir(parents=True, exist_ok=True)
        try:
            video_path = temp_path / "sample.mp4"
            video_path.write_bytes(b"fake-video")
            model = FakeVideoModel(
                [
                    FakeVideoResult(
                        names={0: "lychee"},
                        classes=[0],
                        confidences=[0.91],
                        boxes=[[1, 2, 11, 12]],
                        annotated_frame="frame-0",
                    ),
                    FakeVideoResult(
                        names={0: "branch"},
                        classes=[0],
                        confidences=[0.52],
                        boxes=[[3, 4, 13, 14]],
                        annotated_frame="frame-1",
                    ),
                ]
            )
            fake_cv2 = FakeCv2InferModule(FakeVideoCaptureForInfer(frame_count=2, fps=20.0))
            emitted_frames = []
            service = DetectionService("fake.pt", model_loader=lambda _: model)

            with patch("app.infer.cv2", fake_cv2):
                service.detect_video(
                    video_path,
                    frame_callback=lambda frame_result, total: emitted_frames.append((frame_result, total)),
                )

            self.assertEqual(len(emitted_frames), 2)
            self.assertEqual([total for _, total in emitted_frames], [2, 2])
            self.assertTrue(all(isinstance(frame_result, VideoFrameResult) for frame_result, _ in emitted_frames))
            self.assertEqual(emitted_frames[0][0].frame_index, 0)
            self.assertEqual(emitted_frames[0][0].annotated_frame, "frame-0")
            self.assertEqual(emitted_frames[1][0].frame_index, 1)
            self.assertEqual(emitted_frames[1][0].annotated_frame, "frame-1")
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)


class ExportTests(unittest.TestCase):
    def test_export_detections_csv_writes_expected_rows(self):
        temp_path = Path("tests/.tmp/export_case")
        if temp_path.exists():
            shutil.rmtree(temp_path)
        temp_path.mkdir(parents=True, exist_ok=True)
        try:
            output_path = temp_path / "detections.csv"
            result = DetectionResult(
                image_path=temp_path / "sample.jpg",
                detections=[
                    Detection(label="lychee", confidence=0.91, bbox=(1.0, 2.0, 3.0, 4.0)),
                    Detection(label="branch", confidence=0.55, bbox=(5.0, 6.0, 7.0, 8.0)),
                ],
                counts_by_label={"lychee": 1, "branch": 1},
                total_detections=2,
                annotated_image=b"demo",
            )

            export_detections_csv(result, output_path)

            with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(
                rows,
                [
                    {
                        "image_path": str(temp_path / "sample.jpg"),
                        "label": "lychee",
                        "confidence": "0.9100",
                        "x1": "1.0",
                        "y1": "2.0",
                        "x2": "3.0",
                        "y2": "4.0",
                    },
                    {
                        "image_path": str(temp_path / "sample.jpg"),
                        "label": "branch",
                        "confidence": "0.5500",
                        "x1": "5.0",
                        "y1": "6.0",
                        "x2": "7.0",
                        "y2": "8.0",
                    },
                ],
            )
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)

    def test_export_annotated_image_writes_bytes_payload(self):
        temp_path = Path("tests/.tmp/image_export_case")
        if temp_path.exists():
            shutil.rmtree(temp_path)
        temp_path.mkdir(parents=True, exist_ok=True)
        try:
            output_path = temp_path / "annotated.bin"
            result = DetectionResult(
                image_path=temp_path / "sample.jpg",
                detections=[],
                counts_by_label={},
                total_detections=0,
                annotated_image=b"annotated-image",
            )

            export_annotated_image(result, output_path)

            self.assertEqual(output_path.read_bytes(), b"annotated-image")
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)

    @unittest.skipIf(np is None, "numpy is not available in the current test environment")
    def test_export_annotated_image_writes_png_from_array(self):
        temp_path = Path("tests/.tmp/image_export_png_case")
        if temp_path.exists():
            shutil.rmtree(temp_path)
        temp_path.mkdir(parents=True, exist_ok=True)
        try:
            output_path = temp_path / "annotated.png"
            result = DetectionResult(
                image_path=temp_path / "sample.jpg",
                detections=[],
                counts_by_label={},
                total_detections=0,
                annotated_image=np.zeros((4, 4, 3), dtype=np.uint8),
            )

            export_annotated_image(result, output_path)

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)

    def test_export_video_detections_csv_writes_frame_rows(self):
        from app.export import export_video_detections_csv
        from app.infer import VideoDetectionResult, VideoFrameResult

        temp_path = Path("tests/.tmp/video_export_csv_case")
        if temp_path.exists():
            shutil.rmtree(temp_path)
        temp_path.mkdir(parents=True, exist_ok=True)
        try:
            output_path = temp_path / "video_detections.csv"
            result = VideoDetectionResult(
                video_path=temp_path / "sample.mp4",
                frames=[
                    VideoFrameResult(
                        frame_index=0,
                        timestamp_seconds=0.0,
                        detections=[Detection(label="lychee", confidence=0.91, bbox=(1.0, 2.0, 3.0, 4.0))],
                        counts_by_label={"lychee": 1},
                        total_detections=1,
                        annotated_frame="frame-0",
                    ),
                    VideoFrameResult(
                        frame_index=1,
                        timestamp_seconds=0.5,
                        detections=[Detection(label="branch", confidence=0.55, bbox=(5.0, 6.0, 7.0, 8.0))],
                        counts_by_label={"branch": 1},
                        total_detections=1,
                        annotated_frame="frame-1",
                    ),
                ],
                counts_by_label={"lychee": 1, "branch": 1},
                total_detections=2,
                total_frames=2,
                processed_frames=2,
                fps=2.0,
            )

            export_video_detections_csv(result, output_path)

            with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(
                rows,
                [
                    {
                        "video_path": str(temp_path / "sample.mp4"),
                        "frame_index": "0",
                        "timestamp_seconds": "0.0000",
                        "label": "lychee",
                        "confidence": "0.9100",
                        "x1": "1.0",
                        "y1": "2.0",
                        "x2": "3.0",
                        "y2": "4.0",
                    },
                    {
                        "video_path": str(temp_path / "sample.mp4"),
                        "frame_index": "1",
                        "timestamp_seconds": "0.5000",
                        "label": "branch",
                        "confidence": "0.5500",
                        "x1": "5.0",
                        "y1": "6.0",
                        "x2": "7.0",
                        "y2": "8.0",
                    },
                ],
            )
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)

    def test_export_annotated_video_writes_all_frames(self):
        from app.export import export_annotated_video
        from app.infer import VideoDetectionResult, VideoFrameResult

        temp_path = Path("tests/.tmp/video_export_file_case")
        if temp_path.exists():
            shutil.rmtree(temp_path)
        temp_path.mkdir(parents=True, exist_ok=True)
        try:
            output_path = temp_path / "annotated.mp4"
            result = VideoDetectionResult(
                video_path=temp_path / "sample.mp4",
                frames=[
                    VideoFrameResult(
                        frame_index=0,
                        timestamp_seconds=0.0,
                        detections=[],
                        counts_by_label={},
                        total_detections=0,
                        annotated_frame=[[0, 0], [0, 0]],
                    ),
                    VideoFrameResult(
                        frame_index=1,
                        timestamp_seconds=0.5,
                        detections=[],
                        counts_by_label={},
                        total_detections=0,
                        annotated_frame=[[1, 1], [1, 1]],
                    ),
                ],
                counts_by_label={},
                total_detections=0,
                total_frames=2,
                processed_frames=2,
                fps=2.0,
            )
            fake_cv2 = FakeCv2ExportModule()

            with patch("app.export.cv2", fake_cv2):
                export_annotated_video(result, output_path)

            self.assertEqual(fake_cv2.fourcc_calls, [("m", "p", "4", "v")])
            self.assertEqual(fake_cv2.writer_calls, [(str(output_path), "fourcc", 2.0, (2, 2))])
            self.assertEqual(len(fake_cv2.writers), 1)
            self.assertEqual(fake_cv2.writers[0].frames, [[[0, 0], [0, 0]], [[1, 1], [1, 1]]])
            self.assertTrue(fake_cv2.writers[0].released)
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)


class FakeDetectionService:
    def detect_image(self, _image_path, conf=None, iou=None):
        raise AssertionError("GUI tests should not trigger real detection.")


class StubDetectionService:
    def __init__(self, result):
        self.result = result

    def detect_image(self, _image_path, conf=None, iou=None):
        return self.result


class RecordingDetectionService:
    def __init__(self, results_by_path):
        self.results_by_path = {str(Path(key)): value for key, value in results_by_path.items()}
        self.calls = []

    def detect_image(self, image_path, conf=None, iou=None):
        path = str(Path(image_path))
        self.calls.append(
            {
                "image_path": path,
                "conf": conf,
                "iou": iou,
            }
        )
        return self.results_by_path[path]


class RecordingVideoDetectionService:
    def __init__(self, result):
        self.result = result
        self.video_calls = []

    def detect_image(self, _image_path, conf=None, iou=None):
        raise AssertionError("Video workflow should not call detect_image.")

    def detect_video(self, video_path, conf=None, iou=None, progress_callback=None):
        self.video_calls.append(
            {
                "video_path": str(Path(video_path)),
                "conf": conf,
                "iou": iou,
            }
        )
        if progress_callback is not None:
            progress_callback(self.result.processed_frames, self.result.total_frames)
        return self.result


class RecordingCameraDetectionService:
    def __init__(self, frame_result):
        self.frame_result = frame_result
        self.frame_calls = []

    def detect_image(self, _image_path, conf=None, iou=None):
        raise AssertionError("Camera realtime workflow should not call detect_image.")

    def detect_video(self, _video_path, conf=None, iou=None, progress_callback=None):
        raise AssertionError("Camera realtime workflow should not call detect_video.")

    def detect_frame(self, frame, conf=None, iou=None):
        self.frame_calls.append(
            {
                "frame": frame,
                "conf": conf,
                "iou": iou,
            }
        )
        return self.frame_result


class MainNewEntrypointTests(unittest.TestCase):
    def test_main_new_uses_new_gui_factory(self):
        from app import main_new

        fake_app = type("FakeApp", (), {"exec_": lambda self: 0})()
        with patch("app.main_new.create_new_app", return_value=fake_app) as create_mock:
            exit_code = main_new.main()

        self.assertEqual(exit_code, 0)
        create_mock.assert_called_once()


class LauncherScriptTests(unittest.TestCase):
    def test_start_new_targets_main_new(self):
        script = Path("start_new.bat").read_text(encoding="utf-8")

        self.assertIn("python -m app.main_new", script)
        self.assertIn("CONDA_ENV=", script)


@unittest.skipUnless(QApplication is not None and MainWindow is not None, "PyQt5 GUI dependencies are not available")
class MainWindowLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_path = Path("tests/.tmp/gui_layout_case")
        if self.temp_path.exists():
            shutil.rmtree(self.temp_path)
        self.temp_path.mkdir(parents=True, exist_ok=True)
        self.video_path = self.temp_path / "sample.mp4"
        self.video_path.write_bytes(b"video")
        self.window = MainWindow(FakeDetectionService(), self.temp_path)
        self.window.show()
        self.qt_app.processEvents()

    def tearDown(self):
        self.window.close()
        self.qt_app.processEvents()
        shutil.rmtree(self.temp_path, ignore_errors=True)

    def _child(self, widget_type, object_name):
        widget = self.window.findChild(widget_type, object_name)
        self.assertIsNotNone(widget, f"Missing widget: {object_name}")
        return widget

    def test_has_three_top_mode_entries(self):
        self._child(QWidget, "mode-switcher")
        self._child(QPushButton, "mode-entry-image")
        self._child(QPushButton, "mode-entry-video")
        self._child(QPushButton, "mode-entry-camera")

    def test_status_information_uses_four_spread_metric_cards(self):
        self._child(QWidget, "status-info-bar")
        cards = self.window.findChildren(QFrame, "status-metric-card")
        self.assertEqual(len(cards), 4)
        model_text = self._child(QLabel, "status-model-label").text()
        self.assertIn("yolo26n.pt", model_text)
        self.assertNotIn("best.pt", model_text)
        self.assertIn("\u5f53\u524d\u6587\u4ef6", self._child(QLabel, "status-file-label").text())
        self.assertIn("\u72b6\u6001", self._child(QLabel, "status-state-label").text())
        self.assertIn("\u8017\u65f6", self._child(QLabel, "status-time-label").text())

    def test_left_config_region_uses_large_chinese_controls(self):
        self._child(QWidget, "left-config-panel")
        self.assertEqual(self._child(QLabel, "config-confidence-label").text(), "\u7f6e\u4fe1\u5ea6")
        self.assertEqual(self._child(QLabel, "config-iou-label").text(), "\u4ea4\u5e76\u6bd4")
        self.assertIn(self._child(QLabel, "config-confidence-value").text(), {"0.25", "25%"})
        self.assertIn(self._child(QLabel, "config-iou-value").text(), {"0.45", "45%"})
        self.assertGreaterEqual(self._child(QLabel, "config-confidence-label").minimumHeight(), 0)
        self.assertGreaterEqual(self._child(QSlider, "config-confidence-slider").minimumHeight(), 28)
        self.assertGreaterEqual(self._child(QSlider, "config-iou-slider").minimumHeight(), 28)
        self.assertGreaterEqual(self._child(QCheckBox, "config-show-labels-checkbox").minimumHeight(), 28)
        self.assertGreaterEqual(self._child(QComboBox, "config-export-format").minimumHeight(), 40)

    def test_threshold_sliders_expose_default_numeric_values(self):
        confidence_value = self._child(QLabel, "config-confidence-value")
        iou_value = self._child(QLabel, "config-iou-value")

        self.assertRegex(confidence_value.text(), r"0?\.25|25%")
        self.assertRegex(iou_value.text(), r"0?\.45|45%")
        self.assertRegex(confidence_value.text(), r"\d")
        self.assertRegex(iou_value.text(), r"\d")

    def test_defaults_to_image_mode_and_non_image_controls_are_hidden(self):
        mode_stack = self._child(QStackedWidget, "mode-config-stack")
        self.assertEqual(mode_stack.currentWidget().objectName(), "image-config-panel")
        self.assertTrue(self._child(QPushButton, "mode-entry-image").isChecked())
        self.assertFalse(self._child(QWidget, "video-config-panel").isVisible())
        self.assertFalse(self._child(QWidget, "camera-config-panel").isVisible())

    def test_preview_actions_are_centered(self):
        self._child(QWidget, "preview-action-bar")
        self._child(QWidget, "preview-action-center")
        for name in (
            "preview-action-open",
            "preview-action-detect",
            "preview-action-preview",
            "preview-action-export",
            "preview-action-close-current",
        ):
            self.assertGreaterEqual(self._child(QPushButton, name).minimumHeight(), 48)


@unittest.skipUnless(
    QApplication is not None and MainWindowNew is not None,
    "PyQt5 new GUI dependencies are not available",
)
class MainWindowNewLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_path = Path("tests/.tmp/gui_new_layout_case")
        if self.temp_path.exists():
            shutil.rmtree(self.temp_path)
        self.temp_path.mkdir(parents=True, exist_ok=True)
        self.window = MainWindowNew(FakeDetectionService(), self.temp_path)
        self.window.show()
        self.qt_app.processEvents()

    def tearDown(self):
        self.window.close()
        self.qt_app.processEvents()
        shutil.rmtree(self.temp_path, ignore_errors=True)

    def _child(self, widget_type, object_name):
        widget = self.window.findChild(widget_type, object_name)
        self.assertIsNotNone(widget, f"Missing widget: {object_name}")
        return widget

    def test_new_window_builds_distinct_layout_shell(self):
        self._child(QWidget, "new-root-shell")
        self._child(QWidget, "new-top-bar")
        self._child(QWidget, "new-side-panel")
        self._child(QWidget, "new-preview-panel")
        self._child(QWidget, "new-result-panel")

    def test_new_window_has_same_primary_controls(self):
        self._child(QPushButton, "preview-action-open")
        self._child(QPushButton, "preview-action-detect")
        self._child(QPushButton, "preview-action-export")
        self._child(QComboBox, "config-export-format")

    def test_new_window_starts_in_image_mode(self):
        mode_stack = self._child(QStackedWidget, "mode-config-stack")
        self.assertEqual(self.window.current_mode, "image")
        self.assertEqual(mode_stack.currentWidget().objectName(), "image-config-panel")
        self.assertIn(TEXT["image_config"], self.window.config_title_label.text())

    def test_new_window_uses_new_theme_markers(self):
        stylesheet = self.window.styleSheet()
        self.assertIn("#new-root-shell", stylesheet)
        self.assertIn("QFrame#new-top-bar", stylesheet)


@unittest.skipUnless(
    QApplication is not None and MainWindowNew is not None,
    "PyQt5 new GUI dependencies are not available",
)
class MainWindowNewBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_path = Path("tests/.tmp/gui_new_behavior_case")
        if self.temp_path.exists():
            shutil.rmtree(self.temp_path)
        self.temp_path.mkdir(parents=True, exist_ok=True)
        self.batch_dir = self.temp_path / "batch"
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self.image_one = self.batch_dir / "a.jpg"
        self.image_two = self.batch_dir / "b.png"
        self.image_one.write_bytes(VALID_PNG_BYTES)
        self.image_two.write_bytes(VALID_PNG_BYTES)
        self.window = MainWindowNew(FakeDetectionService(), self.temp_path)
        self.window.show()
        self.qt_app.processEvents()

    def tearDown(self):
        self.window.close()
        self.qt_app.processEvents()
        shutil.rmtree(self.temp_path, ignore_errors=True)

    def test_new_window_batch_selection_updates_current_item(self):
        self.window.image_source_selector.setCurrentIndex(1)

        with patch(
            "app.gui.QFileDialog.getOpenFileNames",
            return_value=([str(self.image_two), str(self.image_one)], "Images (*.png *.jpg *.jpeg *.bmp)"),
        ):
            self.window.select_image()

        self.assertEqual(self.window.batch_image_paths, [self.image_one, self.image_two])
        self.assertEqual(self.window.current_batch_index, 0)
        self.assertEqual(self.window.current_image_path, self.image_one)

    def test_new_window_batch_detection_updates_summary(self):
        result_one = DetectionResult(
            image_path=self.image_one,
            detections=[Detection(label="lychee", confidence=0.9, bbox=(1.0, 2.0, 3.0, 4.0))],
            counts_by_label={"lychee": 1},
            total_detections=1,
            annotated_image=b"",
        )
        result_two = DetectionResult(
            image_path=self.image_two,
            detections=[Detection(label="branch", confidence=0.8, bbox=(5.0, 6.0, 7.0, 8.0))],
            counts_by_label={"branch": 1},
            total_detections=1,
            annotated_image=b"",
        )
        self.window.detection_service = RecordingDetectionService(
            {
                str(self.image_one): result_one,
                str(self.image_two): result_two,
            }
        )
        self.window.image_source_selector.setCurrentIndex(1)
        self.window.batch_image_paths = [self.image_one, self.image_two]
        self.window.current_batch_dir = self.batch_dir

        self.window.run_detection()
        self.qt_app.processEvents()

        self.assertEqual(len(self.window.batch_results), 2)
        self.assertEqual(self.window.detail_list.count(), 2)

    def test_new_window_select_video_updates_status(self):
        video_path = self.temp_path / "sample.mp4"
        video_path.write_bytes(b"video")
        self.window._apply_mode("video")

        with patch("app.gui.QFileDialog.getOpenFileName", return_value=(str(video_path), "Videos (*.mp4)")):
            with patch.object(self.window, "_read_video_preview_pixmap", return_value=QPixmap()):
                self.window.select_video()

        self.assertEqual(self.window.current_video_path, video_path)
        self.assertIn(video_path.name, self.window.status_file_label.text())

    def test_new_window_camera_mode_updates_controls(self):
        self.window._apply_mode("camera")
        self.qt_app.processEvents()

        self.assertEqual(self.window.current_mode, "camera")
        self.assertEqual(self.window.open_action_button.text(), TEXT["open_camera"])
        self._child(QPushButton, "preview-action-close-all")
        self._child(QPushButton, "preview-nav-prev")
        self._child(QPushButton, "preview-nav-next")

    def test_preview_stage_is_square_and_supports_large_preview_dialog(self):
        preview_label = self._child(QLabel, "preview-label")
        preview_scroll_area = self._child(QWidget, "preview-scroll-area")
        self.assertLessEqual(abs(preview_label.minimumWidth() - preview_label.minimumHeight()), 40)
        self.assertGreaterEqual(preview_label.minimumWidth(), 700)
        self.assertTrue(preview_scroll_area.isVisible())
        self.assertEqual(preview_label.property("previewClickAction"), "open-dialog")
        self.assertTrue(hasattr(self.window, "preview_dialog"))
        self.assertIsInstance(self.window.preview_dialog, QDialog)
        self.assertTrue(hasattr(self.window, "open_preview_dialog"))
        self.assertTrue(hasattr(self.window.preview_dialog, "wheelEvent"))
        self.assertGreaterEqual(preview_label.minimumWidth(), 900)

    def test_preview_click_toggles_between_original_and_result(self):
        self.window.original_pixmap = QPixmap(120, 80)
        self.window.annotated_pixmap = QPixmap(120, 80)
        self.window.showing_annotated = True
        self.window.current_result = DetectionResult(
            image_path=self.temp_path / "sample.jpg",
            detections=[],
            counts_by_label={},
            total_detections=0,
            annotated_image=b"",
        )
        self.window._handle_preview_click(None)
        self.assertFalse(self.window.showing_annotated)

    def test_preview_dialog_zoom_state_changes_with_wheel_delta(self):
        self.window.preview_dialog.set_pixmap(QPixmap(200, 120))
        before = self.window.preview_dialog.zoom_factor
        self.window.preview_dialog._zoom_by_delta(120)
        self.assertGreater(self.window.preview_dialog.zoom_factor, before)
        self.window.preview_dialog._zoom_by_delta(-120)
        self.assertLessEqual(self.window.preview_dialog.zoom_factor, 1.0)

    def test_main_preview_zoom_state_changes_with_wheel_delta(self):
        self.window._set_preview(QPixmap(200, 120))
        before = self.window.preview_zoom_factor
        self.window._zoom_preview_by_delta(120)
        self.assertGreater(self.window.preview_zoom_factor, before)
        self.window._zoom_preview_by_delta(-120)
        self.assertLessEqual(self.window.preview_zoom_factor, 1.0)

    def test_main_area_is_two_column_layout(self):
        self._child(QWidget, "main-two-column")
        self._child(QWidget, "left-column-stack")
        self._child(QWidget, "right-preview-column")
        self.assertIsNone(self.window.findChild(QWidget, "result-panel"))

    def test_left_column_stacks_config_and_result_cards(self):
        left_column = self._child(QWidget, "left-column-stack")
        self.assertTrue(left_column.isVisible())
        self._child(QFrame, "left-config-panel")
        self._child(QFrame, "left-result-panel")
        self._child(QFrame, "result-overview-card")
        self._child(QFrame, "result-detail-card")
        self.assertIsNotNone(self.window.findChild(QListWidget, "result-detail-list"))

    def test_right_column_is_single_large_preview_card(self):
        right_column = self._child(QWidget, "right-preview-column")
        preview_panel = self._child(QFrame, "preview-panel")
        self.assertTrue(right_column.isVisible())
        self.assertTrue(preview_panel.isVisible())

    def test_model_status_uses_default_model_filename(self):
        model_text = self._child(QLabel, "status-model-label").text()
        self.assertIn("yolo26n.pt", model_text)
        self.assertNotIn("best.pt", model_text)

    def test_result_detail_list_shows_all_detections(self):
        detections = [
            Detection(label="lychee", confidence=0.80 + (index * 0.01), bbox=(1.0, 2.0, 3.0, 4.0))
            for index in range(11)
        ]
        result = DetectionResult(
            image_path=self.temp_path / "sample.jpg",
            detections=detections,
            counts_by_label={"lychee": 11},
            total_detections=11,
            annotated_image=b"",
        )
        self.window.detection_service = StubDetectionService(result)
        self.window.current_mode = "image"
        self.window.current_image_path = self.temp_path / "sample.jpg"

        self.window.run_detection()
        self.qt_app.processEvents()

        self.assertIn("11", self._child(QLabel, "result-overview-value").text())
        self.assertEqual(self._child(QListWidget, "result-detail-list").count(), 11)

    def test_camera_config_panel_exposes_real_controls(self):
        camera_panel = self._child(QWidget, "camera-config-panel")

        self.assertIsNotNone(camera_panel.findChild(QLabel, "camera-device-label"))
        self.assertIsNotNone(camera_panel.findChild(QComboBox, "camera-device-selector"))
        self.assertIsNotNone(camera_panel.findChild(QLabel, "camera-confidence-label"))
        self.assertIsNotNone(camera_panel.findChild(QSlider, "camera-confidence-slider"))
        self.assertIsNotNone(camera_panel.findChild(QLabel, "camera-confidence-value"))
        self.assertIsNotNone(camera_panel.findChild(QLabel, "camera-iou-label"))
        self.assertIsNotNone(camera_panel.findChild(QSlider, "camera-iou-slider"))
        self.assertIsNotNone(camera_panel.findChild(QLabel, "camera-iou-value"))
        self.assertIsNotNone(camera_panel.findChild(QCheckBox, "camera-show-labels-checkbox"))
        self.assertIsNotNone(camera_panel.findChild(QCheckBox, "camera-save-frames-checkbox"))
        self.assertIsNotNone(camera_panel.findChild(QLabel, "camera-status-label"))
        self.assertNotIn(TEXT["camera_reserved"], camera_panel.findChild(QLabel, "camera-status-label").text())

    def test_video_config_panel_exposes_real_controls(self):
        video_panel = self._child(QWidget, "video-config-panel")

        self.assertIsNotNone(video_panel.findChild(QLabel, "video-source-label"))
        self.assertIsNotNone(video_panel.findChild(QLabel, "video-source-value"))
        self.assertIsNotNone(video_panel.findChild(QSlider, "video-confidence-slider"))
        self.assertIsNotNone(video_panel.findChild(QSlider, "video-iou-slider"))
        self.assertIsNotNone(video_panel.findChild(QCheckBox, "video-show-labels-checkbox"))
        self.assertIsNotNone(video_panel.findChild(QComboBox, "video-export-format"))
        self.assertIsNotNone(video_panel.findChild(QLabel, "video-status-label"))

    def test_switching_to_video_updates_buttons_and_summary(self):
        self.window._apply_mode("video")
        self.qt_app.processEvents()

        mode_stack = self._child(QStackedWidget, "mode-config-stack")
        self.assertEqual(mode_stack.currentWidget().objectName(), "video-config-panel")
        self.assertTrue(self._child(QPushButton, "mode-entry-video").isChecked())
        self.assertIn(TEXT["video_ready"], self._child(QLabel, "status-state-label").text())
        self.assertIn(TEXT["video_overview_ready"], self._child(QLabel, "result-overview-value").text())
        self.assertEqual(self._child(QPushButton, "preview-action-open").text(), TEXT["open"])
        self.assertEqual(self._child(QPushButton, "preview-action-detect").text(), TEXT["detect"])
        self.assertTrue(self._child(QPushButton, "preview-action-export").isVisible())

    def test_switching_to_camera_updates_buttons_and_summary(self):
        self.window._apply_mode("camera")
        self.qt_app.processEvents()

        mode_stack = self._child(QStackedWidget, "mode-config-stack")
        self.assertEqual(mode_stack.currentWidget().objectName(), "camera-config-panel")
        self.assertTrue(self._child(QPushButton, "mode-entry-camera").isChecked())
        self.assertIn("\u6444\u50cf\u5934", self._child(QLabel, "status-state-label").text())
        self.assertIn("\u5b9e\u65f6", self._child(QLabel, "result-overview-value").text())
        self.assertEqual(self._child(QPushButton, "preview-action-open").text(), "\u6253\u5f00\u6444\u50cf\u5934")
        self.assertEqual(self._child(QPushButton, "preview-action-detect").text(), "\u5f00\u59cb\u68c0\u6d4b")
        self.assertFalse(self._child(QPushButton, "preview-action-export").isVisible())
        self.assertFalse(self._child(QPushButton, "preview-action-close-current").isVisible())

    def test_camera_detection_button_toggles_running_state(self):
        self.window._apply_mode("camera")
        frame = np.zeros((24, 32, 3), dtype=np.uint8) if np is not None else None
        fake_capture = FakeCameraCapture(frame=frame)

        with patch("app.gui.cv2", FakeCv2Module(fake_capture)):
            self.window.open_camera()
            self.window.run_detection()
            self.qt_app.processEvents()

        self.assertTrue(self.window.camera_detection_active)
        self.assertEqual(self._child(QPushButton, "preview-action-detect").text(), "\u505c\u6b62\u68c0\u6d4b")
        self.assertIn("\u68c0\u6d4b\u4e2d", self._child(QLabel, "status-state-label").text())
        self.assertIn("\u52a8\u6001\u8ba1\u6570", self._child(QLabel, "result-overview-value").text())

        with patch("app.gui.cv2", FakeCv2Module(fake_capture)):
            self.window.run_detection()
            self.qt_app.processEvents()

        self.assertFalse(self.window.camera_detection_active)
        self.assertEqual(self._child(QPushButton, "preview-action-detect").text(), "\u5f00\u59cb\u68c0\u6d4b")
        self.assertIn("\u5df2\u505c\u6b62", self._child(QLabel, "status-state-label").text())

    @unittest.skipIf(np is None, "numpy not installed")
    def test_open_camera_renders_live_frame_into_preview(self):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[..., 1] = 255
        fake_capture = FakeCameraCapture(frame=frame)

        self.window._apply_mode("camera")
        with patch("app.gui.cv2", FakeCv2Module(fake_capture)):
            self.window.open_camera()

        self.assertTrue(self.window.camera_open)
        self.assertFalse(self.window.current_preview_pixmap.isNull())
        self.assertEqual(self.window.status_file_label.text(), f"{TEXT['current_file']}  {TEXT['camera_device_default']}")

    @unittest.skipIf(np is None, "numpy not installed")
    def test_leaving_camera_mode_releases_capture_device(self):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        fake_capture = FakeCameraCapture(frame=frame)

        self.window._apply_mode("camera")
        with patch("app.gui.cv2", FakeCv2Module(fake_capture)):
            self.window.open_camera()

        self.window._apply_mode("image")

        self.assertTrue(fake_capture.released)
        self.assertFalse(self.window.camera_open)
        self.assertIsNone(self.window.camera_capture)

    def test_video_config_panel_exposes_real_controls(self):
        video_panel = self._child(QWidget, "video-config-panel")

        self.assertIsNotNone(video_panel.findChild(QLabel, "video-source-label"))
        self.assertIsNotNone(video_panel.findChild(QLabel, "video-status-label"))
        self.assertIsNotNone(video_panel.findChild(QSlider, "video-confidence-slider"))
        self.assertIsNotNone(video_panel.findChild(QSlider, "video-iou-slider"))
        self.assertIsNotNone(video_panel.findChild(QComboBox, "video-export-format"))


@unittest.skipUnless(QApplication is not None and MainWindow is not None, "PyQt5 GUI dependencies are not available")
class MainWindowVideoWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_path = Path("tests/.tmp/gui_video_case")
        if self.temp_path.exists():
            shutil.rmtree(self.temp_path)
        self.temp_path.mkdir(parents=True, exist_ok=True)
        self.video_path = self.temp_path / "sample.mp4"
        self.video_path.write_bytes(b"video")
        self.window = MainWindow(FakeDetectionService(), self.temp_path)
        self.window.show()
        self.qt_app.processEvents()

    def tearDown(self):
        self.window.close()
        self.qt_app.processEvents()
        shutil.rmtree(self.temp_path, ignore_errors=True)

    def _child(self, widget_type, object_name):
        widget = self.window.findChild(widget_type, object_name)
        self.assertIsNotNone(widget, f"Missing widget: {object_name}")
        return widget

    def _video_result(self):
        first_raw = QPixmap(12, 12)
        first_raw.fill()
        second_raw = QPixmap(18, 18)
        second_raw.fill()
        return VideoDetectionResult(
            video_path=self.video_path,
            frames=[
                VideoFrameResult(
                    frame_index=0,
                    timestamp_seconds=0.0,
                    detections=[Detection(label="lychee", confidence=0.91, bbox=(1.0, 2.0, 3.0, 4.0))],
                    counts_by_label={"lychee": 1},
                    total_detections=1,
                    annotated_frame=VALID_PNG_BYTES,
                    raw_frame=first_raw,
                ),
                VideoFrameResult(
                    frame_index=1,
                    timestamp_seconds=0.5,
                    detections=[Detection(label="branch", confidence=0.55, bbox=(5.0, 6.0, 7.0, 8.0))],
                    counts_by_label={"branch": 1},
                    total_detections=1,
                    annotated_frame=VALID_PNG_BYTES,
                    raw_frame=second_raw,
                ),
            ],
            counts_by_label={"lychee": 1, "branch": 1},
            total_detections=2,
            total_frames=2,
            processed_frames=2,
            fps=2.0,
        )

    def test_open_current_source_uses_video_picker_in_video_mode(self):
        self.window._apply_mode("video")

        with patch(
            "app.gui.QFileDialog.getOpenFileName",
            return_value=(str(self.video_path), "Videos (*.mp4 *.avi *.mov *.mkv)"),
        ):
            self.window.open_current_source()

        self.assertEqual(self.window.current_video_path, self.video_path)
        self.assertIsNone(self.window.current_video_result)
        self.assertIn(self.video_path.name, self.window.status_file_label.text())
        self.assertIn(TEXT["video_loaded"], self.window.overview_value_label.text())

    def test_run_detection_in_video_mode_uses_worker_start_path(self):
        self.window._apply_mode("video")
        self.window.current_video_path = self.video_path
        started = []

        self.window._start_video_detection = lambda: started.append("started")

        self.window.run_detection()

        self.assertEqual(started, ["started"])

    def test_click_video_toggles_pause_and_resume(self):
        self.window._apply_mode("video")
        self.window._render_video_result(self._video_result(), elapsed=1.23)

        self.window._handle_preview_click(None)

        self.assertTrue(self.window.video_paused)
        self.assertFalse(self.window.video_playback_timer.isActive())

        self.window._handle_preview_click(None)

        self.assertFalse(self.window.video_paused)
        self.assertTrue(self.window.video_playback_timer.isActive())

    def test_click_video_during_processing_pauses_preview_without_prompt(self):
        self.window._apply_mode("video")
        self.window.current_video_path = self.video_path
        self.window.video_processing = True

        with patch("app.gui.QMessageBox.information") as info_mock:
            self.window._handle_preview_click(None)

        self.assertTrue(self.window.video_paused)
        info_mock.assert_not_called()

    def test_video_toggle_button_is_blocked_until_video_paused(self):
        self.window._apply_mode("video")
        self.window._render_video_result(self._video_result(), elapsed=1.23)
        toggle_button = self._child(QPushButton, "preview-action-video-original")
        hidden_result_button = self._child(QPushButton, "preview-action-video-result")

        self.assertFalse(toggle_button.isEnabled())
        self.assertFalse(hidden_result_button.isVisible())
        self.assertTrue(self.window.video_showing_annotated)
        self.assertEqual(toggle_button.text(), TEXT["video_original"])

        self.window.toggle_video_preview_frame()

        self.assertTrue(self.window.video_showing_annotated)

        self.window._handle_preview_click(None)

        self.assertTrue(toggle_button.isEnabled())

        self.window.toggle_video_preview_frame()
        self.assertFalse(self.window.video_showing_annotated)
        self.assertEqual(toggle_button.text(), TEXT["video_result"])

        self.window.toggle_video_preview_frame()
        self.assertTrue(self.window.video_showing_annotated)
        self.assertEqual(toggle_button.text(), TEXT["video_original"])

    def test_restart_video_resets_to_first_frame_and_can_play_again(self):
        self.window._apply_mode("video")
        result = self._video_result()
        self.window._render_video_result(result, elapsed=1.23)
        self.window._show_video_frame(1)
        self.window.video_next_frame_index = len(result.frames)
        before_cache_key = self.window.current_preview_pixmap.cacheKey()

        self.window.restart_video_playback()

        self.assertEqual(self.window.video_current_frame_index, 0)
        self.assertEqual(self.window.video_next_frame_index, 1)
        self.assertFalse(self.window.video_paused)
        self.assertTrue(self.window.video_playback_timer.isActive())
        self.assertNotEqual(self.window.current_preview_pixmap.cacheKey(), before_cache_key)

    def test_render_video_detection_result_updates_preview_and_aggregated_results(self):
        self.window._apply_mode("video")
        result = self._video_result()

        self.window._render_video_result(result, elapsed=1.23)
        self.qt_app.processEvents()

        self.assertEqual(self.window.current_video_result, result)
        self.assertEqual(self.window.current_result, result.frames[-1])
        self.assertFalse(self.window.current_preview_pixmap.isNull())
        self.assertIn(self.video_path.name, self.window.status_file_label.text())
        self.assertIn("2", self.window.overview_value_label.text())
        self.assertEqual(self.window.detail_list.count(), 2)
        self.assertIn("1.23s", self.window.status_time_label.text())

    def test_export_by_selection_exports_video_csv_and_video_file(self):
        self.window._apply_mode("video")
        self.window.current_video_path = self.video_path
        self.window.current_video_result = self._video_result()
        self.window.video_export_format_selector.setCurrentText(TEXT["export_both"])

        def _write_csv(result, output_path):
            Path(output_path).write_text("csv", encoding="utf-8")
            return Path(output_path)

        def _write_video(result, output_path):
            Path(output_path).write_bytes(b"video")
            return Path(output_path)

        with patch("app.gui.QMessageBox.information"), patch(
            "app.gui.export_video_detections_csv",
            side_effect=_write_csv,
        ), patch(
            "app.gui.export_annotated_video",
            side_effect=_write_video,
        ):
            self.window.export_by_selection()

        csv_path = self.temp_path / f"{self.video_path.stem}_detections.csv"
        video_output_path = self.temp_path / f"{self.video_path.stem}_annotated.mp4"
        self.assertTrue(csv_path.exists())
        self.assertTrue(video_output_path.exists())


@unittest.skipUnless(QApplication is not None and MainWindow is not None, "PyQt5 GUI dependencies are not available")
@unittest.skipIf(np is None, "numpy not installed")
class MainWindowCameraWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_path = Path("tests/.tmp/gui_camera_case")
        if self.temp_path.exists():
            shutil.rmtree(self.temp_path)
        self.temp_path.mkdir(parents=True, exist_ok=True)
        self.window = MainWindow(FakeDetectionService(), self.temp_path)
        self.window.show()
        self.qt_app.processEvents()

    def tearDown(self):
        self.window.close()
        self.qt_app.processEvents()
        shutil.rmtree(self.temp_path, ignore_errors=True)

    def _child(self, widget_type, object_name):
        widget = self.window.findChild(widget_type, object_name)
        self.assertIsNotNone(widget, f"Missing widget: {object_name}")
        return widget

    def _camera_frame_result(self):
        return CameraFrameResult(
            detections=[
                Detection(label="lychee", confidence=0.91, bbox=(1.0, 2.0, 3.0, 4.0)),
                Detection(label="branch", confidence=0.55, bbox=(5.0, 6.0, 7.0, 8.0)),
            ],
            counts_by_label={"lychee": 1, "branch": 1},
            total_detections=2,
            annotated_frame=VALID_PNG_BYTES,
        )

    def test_open_camera_starts_preview_without_starting_realtime_detection(self):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[..., 1] = 255
        fake_capture = FakeCameraCapture(frame=frame)
        detect_button = self._child(QPushButton, "preview-action-detect")

        self.window._apply_mode("camera")
        with patch("app.gui.cv2", FakeCv2Module(fake_capture)):
            self.window.open_camera()

        self.assertTrue(self.window.camera_open)
        self.assertFalse(self.window.camera_detection_active)
        self.assertEqual(detect_button.text(), TEXT["detect"])
        self.assertFalse(self.window.current_preview_pixmap.isNull())

    def test_open_camera_tries_multiple_backends_and_uses_first_capture_with_valid_frame(self):
        invalid_capture = SequencedCameraCapture([(True, None)])
        valid_frame = np.zeros((24, 32, 3), dtype=np.uint8)
        valid_frame[..., 1] = 255
        valid_capture = SequencedCameraCapture([(True, valid_frame)])
        fake_cv2 = SequencedCameraCv2Module([invalid_capture, valid_capture])

        self.window._apply_mode("camera")
        with patch("app.gui.cv2", fake_cv2):
            self.window.open_camera()

        self.assertEqual(
            fake_cv2.video_capture_calls,
            [
                (0, fake_cv2.CAP_DSHOW),
                (0, fake_cv2.CAP_MSMF),
            ],
        )
        self.assertIs(self.window.camera_capture, valid_capture)
        self.assertTrue(self.window.camera_open)
        self.assertFalse(self.window.current_preview_pixmap.isNull())
        self.assertTrue(invalid_capture.released)

    def test_active_camera_frame_uses_detection_service_and_updates_result_view(self):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[..., 1] = 255
        fake_capture = FakeCameraCapture(frame=frame)
        self.window.detection_service = RecordingCameraDetectionService(self._camera_frame_result())
        self.window._apply_mode("camera")

        self._child(QSlider, "camera-confidence-slider").setValue(33)
        self._child(QSlider, "camera-iou-slider").setValue(61)

        with patch("app.gui.cv2", FakeCv2Module(fake_capture)):
            self.window.open_camera()
            self.window.run_detection()
            self.window._update_camera_preview()

        self.assertEqual(len(self.window.detection_service.frame_calls), 1)
        self.assertIs(self.window.detection_service.frame_calls[0]["frame"], frame)
        self.assertEqual(
            self.window.detection_service.frame_calls[0],
            {
                "frame": frame,
                "conf": 0.33,
                "iou": 0.61,
            },
        )
        self.assertIn("2", self.window.overview_value_label.text())
        self.assertEqual(self.window.detail_list.count(), 2)
        self.assertNotEqual(self.window.status_time_label.text(), f"{TEXT['elapsed']}  0.00s")

    def test_camera_preview_skips_invalid_frames_until_a_valid_frame_arrives(self):
        valid_frame = np.zeros((24, 32, 3), dtype=np.uint8)
        valid_frame[..., 2] = 255
        fake_capture = SequencedCameraCapture(
            [
                (False, None),
                (True, None),
                (True, np.zeros((0, 0, 3), dtype=np.uint8)),
                (True, valid_frame),
            ]
        )
        self.window._apply_mode("camera")
        self.window.camera_capture = fake_capture
        self.window.camera_open = True

        converted_frames = []

        def record_pixmap(frame):
            converted_frames.append(frame)
            return QPixmap(10, 10)

        with patch.object(self.window, "_camera_frame_to_qpixmap", side_effect=record_pixmap):
            self.window._update_camera_preview()
            self.window._update_camera_preview()
            self.window._update_camera_preview()

            self.assertEqual(converted_frames, [])
            self.assertTrue(self.window.current_preview_pixmap.isNull())

            self.window._update_camera_preview()

        self.assertEqual(converted_frames, [valid_frame])
        self.assertFalse(self.window.current_preview_pixmap.isNull())

    def test_camera_preview_switches_between_raw_and_annotated_frames_by_detection_state(self):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[..., 1] = 255
        fake_capture = FakeCameraCapture(frame=frame)
        self.window.detection_service = RecordingCameraDetectionService(self._camera_frame_result())
        self.window._apply_mode("camera")

        with patch("app.gui.cv2", FakeCv2Module(fake_capture)):
            self.window.open_camera()

            self.assertFalse(self.window.showing_annotated)
            self.assertTrue(self.window.annotated_pixmap.isNull())

            self.window.run_detection()
            self.window._update_camera_preview()

        self.assertTrue(self.window.showing_annotated)
        self.assertFalse(self.window.annotated_pixmap.isNull())
        self.assertEqual(
            self.window.current_preview_pixmap.cacheKey(),
            self.window.annotated_pixmap.cacheKey(),
        )

    def test_camera_save_frames_checkbox_writes_annotated_frame_file_when_enabled(self):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[..., 1] = 255
        fake_capture = FakeCameraCapture(frame=frame)
        self.window.detection_service = RecordingCameraDetectionService(self._camera_frame_result())
        self.window._apply_mode("camera")
        self._child(QCheckBox, "camera-save-frames-checkbox").setChecked(True)

        with patch("app.gui.cv2", FakeCv2Module(fake_capture)):
            self.window.open_camera()
            self.window.run_detection()
            self.window._update_camera_preview()

        saved_images = list((self.temp_path / "camera_frames" / "device_0").glob("*.jpg"))
        self.assertGreaterEqual(len(saved_images), 1)


@unittest.skipUnless(QApplication is not None and MainWindow is not None, "PyQt5 GUI dependencies are not available")
class MainWindowBatchWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_path = Path("tests/.tmp/gui_batch_case")
        if self.temp_path.exists():
            shutil.rmtree(self.temp_path)
        self.temp_path.mkdir(parents=True, exist_ok=True)
        self.batch_dir = self.temp_path / "batch"
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self.image_one = self.batch_dir / "a.jpg"
        self.image_two = self.batch_dir / "b.png"
        self.non_image = self.batch_dir / "note.txt"
        self.image_one.write_bytes(b"image-a")
        self.image_two.write_bytes(b"image-b")
        self.non_image.write_text("ignore", encoding="utf-8")
        self.window = MainWindow(FakeDetectionService(), self.temp_path)
        self.window.show()
        self.qt_app.processEvents()

    def tearDown(self):
        self.window.close()
        self.qt_app.processEvents()
        shutil.rmtree(self.temp_path, ignore_errors=True)

    def _child(self, widget_type, object_name):
        widget = self.window.findChild(widget_type, object_name)
        self.assertIsNotNone(widget, f"Missing widget: {object_name}")
        return widget

    def test_select_image_uses_file_picker_in_batch_mode(self):
        self.window.image_source_selector.setCurrentIndex(1)

        with patch(
            "app.gui.QFileDialog.getOpenFileNames",
            return_value=([str(self.image_two), str(self.image_one)], "Images (*.png *.jpg *.jpeg *.bmp)"),
        ):
            self.window.select_image()

        self.assertEqual(self.window.current_batch_dir, self.batch_dir)
        self.assertEqual(self.window.batch_image_paths, [self.image_one, self.image_two])
        self.assertEqual(self.window.current_batch_index, 0)
        self.assertEqual(self.window.current_image_path, self.image_one)
        self.assertIn("2", self.window.overview_value_label.text())
        self.assertIn(self.image_one.name, self.window.status_file_label.text())
        self.assertIn("1/2", self.window.status_file_label.text())

    def test_switching_from_single_image_to_batch_preserves_single_image_state_for_restore(self):
        result = DetectionResult(
            image_path=self.image_one,
            detections=[Detection(label="lychee", confidence=0.91, bbox=(1.0, 2.0, 3.0, 4.0))],
            counts_by_label={"lychee": 1},
            total_detections=1,
            annotated_image=b"annotated-a",
        )
        self.window.current_image_path = self.image_one
        self.window.current_result = result
        self.window.original_pixmap = QPixmap(120, 80)
        self.window.annotated_pixmap = QPixmap(120, 80)
        self.window.showing_annotated = True
        self.window._set_preview(self.window.annotated_pixmap)
        self.window.status_file_label.setText(f"{TEXT['current_file']}  {self.image_one.name}")

        self.window.image_source_selector.setCurrentIndex(1)
        self.qt_app.processEvents()

        self.assertIn(TEXT["need_batch_body"], self.window.overview_value_label.text())
        self.assertTrue(self.window.current_preview_pixmap.isNull())

        self.window.image_source_selector.setCurrentIndex(0)
        self.qt_app.processEvents()

        self.assertEqual(self.window.current_image_path, self.image_one)
        self.assertIs(self.window.current_result, result)
        self.assertTrue(self.window.showing_annotated)
        self.assertFalse(self.window.current_preview_pixmap.isNull())
        self.assertEqual(
            self.window.current_preview_pixmap.cacheKey(),
            self.window.annotated_pixmap.cacheKey(),
        )

    def test_switching_to_video_preserves_image_state_and_restores_preview_when_returning(self):
        result = DetectionResult(
            image_path=self.image_one,
            detections=[Detection(label="lychee", confidence=0.91, bbox=(1.0, 2.0, 3.0, 4.0))],
            counts_by_label={"lychee": 1},
            total_detections=1,
            annotated_image=b"annotated-a",
        )
        self.window.current_image_path = self.image_one
        self.window.current_result = result
        self.window.original_pixmap = QPixmap(120, 80)
        self.window.annotated_pixmap = QPixmap(120, 80)
        self.window.showing_annotated = True
        self.window._set_preview(self.window.annotated_pixmap)
        self.window.status_file_label.setText(f"{TEXT['current_file']}  {self.image_one.name}")

        self.window._apply_mode("video")
        self.qt_app.processEvents()

        self.assertEqual(self.window.current_image_path, self.image_one)
        self.assertIs(self.window.current_result, result)
        self.assertFalse(self.window.original_pixmap.isNull())
        self.assertFalse(self.window.annotated_pixmap.isNull())
        self.assertTrue(self.window.current_preview_pixmap.isNull())
        self.assertEqual(self.window.image_label.text(), TEXT["preview_empty"])
        self.assertIn(TEXT["need_video_body"], self.window.overview_value_label.text())

        self.window._apply_mode("image")
        self.qt_app.processEvents()

        self.assertEqual(self.window.current_image_path, self.image_one)
        self.assertIs(self.window.current_result, result)
        self.assertTrue(self.window.showing_annotated)
        self.assertFalse(self.window.current_preview_pixmap.isNull())
        self.assertEqual(
            self.window.current_preview_pixmap.cacheKey(),
            self.window.annotated_pixmap.cacheKey(),
        )
        self.assertIn(self.image_one.name, self.window.status_file_label.text())

    def test_select_video_uses_file_picker_and_updates_status(self):
        video_path = self.temp_path / "sample.mp4"
        video_path.write_bytes(b"video")
        self.window._apply_mode("video")

        with patch("app.gui.QFileDialog.getOpenFileName", return_value=(str(video_path), "Videos (*.mp4)")):
            with patch.object(self.window, "_read_video_preview_pixmap", return_value=QPixmap()):
                self.window.select_video()

        self.assertEqual(self.window.current_video_path, video_path)
        self.assertIsNone(self.window.current_video_result)
        self.assertEqual(self.window.video_source_value.text(), video_path.name)
        self.assertIn(video_path.name, self.window.status_file_label.text())
        self.assertIn(TEXT["video_loaded"], self.window.overview_value_label.text())

    def test_export_by_selection_exports_video_csv_and_annotated_video(self):
        from app.infer import VideoDetectionResult, VideoFrameResult

        video_path = self.temp_path / "sample.mp4"
        video_path.write_bytes(b"video")
        self.window._apply_mode("video")
        self.window.current_video_path = video_path
        self.window.current_video_result = VideoDetectionResult(
            video_path=video_path,
            frames=[
                VideoFrameResult(
                    frame_index=0,
                    timestamp_seconds=0.0,
                    detections=[Detection(label="lychee", confidence=0.9, bbox=(1.0, 2.0, 3.0, 4.0))],
                    counts_by_label={"lychee": 1},
                    total_detections=1,
                    annotated_frame=[[0, 0], [0, 0]],
                )
            ],
            counts_by_label={"lychee": 1},
            total_detections=1,
            total_frames=1,
            processed_frames=1,
            fps=1.0,
        )
        self.window.video_export_format_selector.setCurrentText(TEXT["export_both"])

        with patch("app.gui.export_annotated_video") as export_video_mock:
            with patch("app.gui.QMessageBox.information"):
                self.window.export_by_selection()

        csv_path = self.temp_path / "sample_detections.csv"
        video_output = self.temp_path / "sample_annotated.mp4"
        self.assertTrue(csv_path.exists())
        export_video_mock.assert_called_once_with(self.window.current_video_result, video_output)

    def test_close_current_clears_single_image_state_and_prevents_restore_after_mode_switch(self):
        close_current_button = self._child(QPushButton, "preview-action-close-current")
        result = DetectionResult(
            image_path=self.image_one,
            detections=[Detection(label="lychee", confidence=0.91, bbox=(1.0, 2.0, 3.0, 4.0))],
            counts_by_label={"lychee": 1},
            total_detections=1,
            annotated_image=b"annotated-a",
        )
        self.window.current_image_path = self.image_one
        self.window.current_result = result
        self.window.original_pixmap = QPixmap(120, 80)
        self.window.annotated_pixmap = QPixmap(120, 80)
        self.window.showing_annotated = True
        self.window._set_preview(self.window.annotated_pixmap)

        close_current_button.click()
        self.qt_app.processEvents()

        self.assertIsNone(self.window.single_image_path)
        self.assertIsNone(self.window.single_result)
        self.assertIsNone(self.window.current_image_path)
        self.assertIsNone(self.window.current_result)
        self.assertTrue(self.window.single_original_pixmap.isNull())
        self.assertTrue(self.window.single_annotated_pixmap.isNull())
        self.assertTrue(self.window.original_pixmap.isNull())
        self.assertTrue(self.window.annotated_pixmap.isNull())
        self.assertTrue(self.window.current_preview_pixmap.isNull())
        self.assertEqual(self.window.image_label.text(), TEXT["preview_empty"])
        self.assertEqual(self.window.overview_value_label.text(), TEXT["no_result"])
        self.assertFalse(close_current_button.isVisible())

        self.window._apply_mode("video")
        self.qt_app.processEvents()
        self.window._apply_mode("image")
        self.qt_app.processEvents()

        self.assertIsNone(self.window.single_image_path)
        self.assertIsNone(self.window.current_image_path)
        self.assertIsNone(self.window.single_result)
        self.assertIsNone(self.window.current_result)
        self.assertTrue(self.window.current_preview_pixmap.isNull())
        self.assertEqual(self.window.image_label.text(), TEXT["preview_empty"])
        self.assertEqual(self.window.overview_value_label.text(), TEXT["no_result"])

    def test_close_current_in_batch_removes_current_item_and_shows_next_image(self):
        close_current_button = self._child(QPushButton, "preview-action-close-current")
        result_one = DetectionResult(
            image_path=self.image_one,
            detections=[Detection(label="lychee", confidence=0.91, bbox=(1.0, 2.0, 3.0, 4.0))],
            counts_by_label={"lychee": 1},
            total_detections=1,
            annotated_image=b"annotated-a",
        )
        result_two = DetectionResult(
            image_path=self.image_two,
            detections=[Detection(label="branch", confidence=0.55, bbox=(9.0, 10.0, 11.0, 12.0))],
            counts_by_label={"branch": 1},
            total_detections=1,
            annotated_image=b"annotated-b",
        )
        self.window.image_source_selector.setCurrentIndex(1)
        self.window.current_batch_dir = self.batch_dir
        self.window.batch_image_paths = [self.image_one, self.image_two]
        self.window.batch_results = [result_one, result_two]
        self.window._show_batch_item(0, prefer_annotated=True)
        self.qt_app.processEvents()

        close_current_button.click()
        self.qt_app.processEvents()

        self.assertEqual(self.window.batch_image_paths, [self.image_two])
        self.assertEqual(self.window.batch_results, [result_two])
        self.assertEqual(self.window.current_batch_index, 0)
        self.assertEqual(self.window.current_image_path, self.image_two)
        self.assertIs(self.window.current_result, result_two)
        self.assertIn(self.image_two.name, self.window.status_file_label.text())
        self.assertIn("1/1", self.window.status_file_label.text())

    def test_close_current_in_batch_removes_last_item_and_selects_new_last_image(self):
        close_current_button = self._child(QPushButton, "preview-action-close-current")
        result_one = DetectionResult(
            image_path=self.image_one,
            detections=[Detection(label="lychee", confidence=0.91, bbox=(1.0, 2.0, 3.0, 4.0))],
            counts_by_label={"lychee": 1},
            total_detections=1,
            annotated_image=b"annotated-a",
        )
        result_two = DetectionResult(
            image_path=self.image_two,
            detections=[Detection(label="branch", confidence=0.55, bbox=(9.0, 10.0, 11.0, 12.0))],
            counts_by_label={"branch": 1},
            total_detections=1,
            annotated_image=b"annotated-b",
        )
        self.window.image_source_selector.setCurrentIndex(1)
        self.window.current_batch_dir = self.batch_dir
        self.window.batch_image_paths = [self.image_one, self.image_two]
        self.window.batch_results = [result_one, result_two]
        self.window._show_batch_item(1, prefer_annotated=True)
        self.qt_app.processEvents()

        close_current_button.click()
        self.qt_app.processEvents()

        self.assertEqual(self.window.batch_image_paths, [self.image_one])
        self.assertEqual(self.window.batch_results, [result_one])
        self.assertEqual(self.window.current_batch_index, 0)
        self.assertEqual(self.window.current_image_path, self.image_one)
        self.assertIs(self.window.current_result, result_one)
        self.assertIn(self.image_one.name, self.window.status_file_label.text())
        self.assertIn("1/1", self.window.status_file_label.text())

    def test_close_all_is_batch_only_and_clears_batch_state(self):
        close_all_button = self._child(QPushButton, "preview-action-close-all")
        self.assertFalse(close_all_button.isVisible())

        result_one = DetectionResult(
            image_path=self.image_one,
            detections=[Detection(label="lychee", confidence=0.91, bbox=(1.0, 2.0, 3.0, 4.0))],
            counts_by_label={"lychee": 1},
            total_detections=1,
            annotated_image=b"annotated-a",
        )
        result_two = DetectionResult(
            image_path=self.image_two,
            detections=[Detection(label="branch", confidence=0.55, bbox=(9.0, 10.0, 11.0, 12.0))],
            counts_by_label={"branch": 1},
            total_detections=1,
            annotated_image=b"annotated-b",
        )
        self.window.image_source_selector.setCurrentIndex(1)
        self.window.current_batch_dir = self.batch_dir
        self.window.batch_image_paths = [self.image_one, self.image_two]
        self.window.batch_results = [result_one, result_two]
        self.window._show_batch_item(1, prefer_annotated=True)
        self.qt_app.processEvents()

        self.assertTrue(close_all_button.isVisible())

        close_all_button.click()
        self.qt_app.processEvents()

        self.assertEqual(self.window.batch_image_paths, [])
        self.assertEqual(self.window.batch_results, [])
        self.assertIsNone(self.window.current_batch_dir)
        self.assertIsNone(self.window.current_batch_index)
        self.assertIsNone(self.window.current_image_path)
        self.assertIsNone(self.window.current_result)
        self.assertTrue(self.window.current_preview_pixmap.isNull())
        self.assertEqual(self.window.image_label.text(), TEXT["preview_empty"])
        self.assertIn(TEXT["need_batch_body"], self.window.overview_value_label.text())
        self.assertFalse(close_all_button.isVisible())

    def test_run_detection_processes_batch_images_and_updates_summary(self):
        result_one = DetectionResult(
            image_path=self.image_one,
            detections=[Detection(label="lychee", confidence=0.91, bbox=(1.0, 2.0, 3.0, 4.0))],
            counts_by_label={"lychee": 1},
            total_detections=1,
            annotated_image=b"",
        )
        result_two = DetectionResult(
            image_path=self.image_two,
            detections=[
                Detection(label="lychee", confidence=0.85, bbox=(5.0, 6.0, 7.0, 8.0)),
                Detection(label="branch", confidence=0.55, bbox=(9.0, 10.0, 11.0, 12.0)),
            ],
            counts_by_label={"lychee": 1, "branch": 1},
            total_detections=2,
            annotated_image=b"",
        )
        self.window.detection_service = RecordingDetectionService(
            {
                str(self.image_one): result_one,
                str(self.image_two): result_two,
            }
        )
        self.window.image_source_selector.setCurrentIndex(1)
        self.window.batch_image_paths = [self.image_one, self.image_two]
        self.window.current_batch_dir = self.batch_dir
        self.window.conf_slider.setValue(33)
        self.window.iou_slider.setValue(61)

        self.window.run_detection()
        self.qt_app.processEvents()

        self.assertEqual(
            self.window.detection_service.calls,
            [
                {"image_path": str(self.image_one), "conf": 0.33, "iou": 0.61},
                {"image_path": str(self.image_two), "conf": 0.33, "iou": 0.61},
            ],
        )
        self.assertEqual(len(self.window.batch_results), 2)
        self.assertEqual(self.window.current_result.image_path, self.image_two)
        self.assertEqual(self.window.current_batch_index, 1)
        self.assertIn("2", self.window.overview_value_label.text())
        self.assertIn("3", self.window.overview_value_label.text())
        self.assertEqual(self.window.detail_list.count(), 3)

    def test_batch_navigation_switches_current_preview_image(self):
        result_one = DetectionResult(
            image_path=self.image_one,
            detections=[Detection(label="lychee", confidence=0.91, bbox=(1.0, 2.0, 3.0, 4.0))],
            counts_by_label={"lychee": 1},
            total_detections=1,
            annotated_image=b"annotated-a",
        )
        result_two = DetectionResult(
            image_path=self.image_two,
            detections=[Detection(label="branch", confidence=0.55, bbox=(9.0, 10.0, 11.0, 12.0))],
            counts_by_label={"branch": 1},
            total_detections=1,
            annotated_image=b"annotated-b",
        )
        self.window.image_source_selector.setCurrentIndex(1)
        self.window.current_batch_dir = self.batch_dir
        self.window.batch_image_paths = [self.image_one, self.image_two]
        self.window.batch_results = [result_one, result_two]

        self.window._show_batch_item(0, prefer_annotated=True)
        self.qt_app.processEvents()

        self.assertEqual(self.window.current_batch_index, 0)
        self.assertEqual(self.window.current_result.image_path, self.image_one)
        self.assertFalse(self.window.preview_nav_prev.isEnabled())
        self.assertTrue(self.window.preview_nav_next.isEnabled())

        self.window.show_next_batch_image()
        self.qt_app.processEvents()

        self.assertEqual(self.window.current_batch_index, 1)
        self.assertEqual(self.window.current_result.image_path, self.image_two)
        self.assertTrue(self.window.preview_nav_prev.isEnabled())
        self.assertFalse(self.window.preview_nav_next.isEnabled())
        self.assertIn(self.image_two.name, self.window.status_file_label.text())

    def test_export_by_selection_exports_batch_csv_and_annotated_images(self):
        result_one = DetectionResult(
            image_path=self.image_one,
            detections=[Detection(label="lychee", confidence=0.91, bbox=(1.0, 2.0, 3.0, 4.0))],
            counts_by_label={"lychee": 1},
            total_detections=1,
            annotated_image=b"annotated-a",
        )
        result_two = DetectionResult(
            image_path=self.image_two,
            detections=[Detection(label="branch", confidence=0.55, bbox=(9.0, 10.0, 11.0, 12.0))],
            counts_by_label={"branch": 1},
            total_detections=1,
            annotated_image=b"annotated-b",
        )
        self.window.image_source_selector.setCurrentIndex(1)
        self.window.current_batch_dir = self.batch_dir
        self.window.batch_image_paths = [self.image_one, self.image_two]
        self.window.batch_results = [result_one, result_two]
        self.window.current_result = result_two
        self.window.export_format_selector.setCurrentText(TEXT["export_both"])

        with patch("app.gui.QMessageBox.information"):
            self.window.export_by_selection()

        csv_path = self.temp_path / f"{self.batch_dir.name}_detections.csv"
        image_dir = self.temp_path / f"{self.batch_dir.name}_annotated"
        self.assertTrue(csv_path.exists())
        self.assertTrue((image_dir / "a_annotated.jpg").exists())
        self.assertTrue((image_dir / "b_annotated.png").exists())
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
