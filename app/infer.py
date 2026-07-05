from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - depends on local env
    cv2 = None

try:
    from ultralytics import YOLO
except ModuleNotFoundError:  # pragma: no cover - depends on local env
    YOLO = None


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectionResult:
    image_path: Path
    detections: list[Detection]
    counts_by_label: dict[str, int]
    total_detections: int
    annotated_image: object


@dataclass(frozen=True)
class VideoFrameResult:
    frame_index: int
    timestamp_seconds: float
    detections: list[Detection]
    counts_by_label: dict[str, int]
    total_detections: int
    annotated_frame: object
    raw_frame: object | None = None


@dataclass(frozen=True)
class CameraFrameResult:
    detections: list[Detection]
    counts_by_label: dict[str, int]
    total_detections: int
    annotated_frame: object

    @property
    def annotated_image(self) -> object:
        return self.annotated_frame


@dataclass(frozen=True)
class VideoDetectionResult:
    video_path: Path
    frames: list[VideoFrameResult]
    counts_by_label: dict[str, int]
    total_detections: int
    total_frames: int | None
    processed_frames: int
    fps: float | None


class DetectionService:
    def __init__(self, model_path: str | Path, model_loader: Callable[[str | Path], object] | None = None):
        self.model_path = Path(model_path)
        if model_loader is None and YOLO is None:
            raise ModuleNotFoundError("ultralytics is required when no custom model_loader is provided.")
        self._model_loader = model_loader or YOLO
        self._model = self._model_loader(str(self.model_path))

    def _build_detections(self, result: object) -> list[Detection]:
        detections: list[Detection] = []
        for cls_id, confidence, bbox in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy):
            label = result.names[int(cls_id)]
            detections.append(
                Detection(
                    label=label,
                    confidence=float(confidence),
                    bbox=tuple(float(value) for value in bbox),
                )
            )
        return detections

    def _build_predict_kwargs(
        self,
        source: object,
        conf: float | None = None,
        iou: float | None = None,
        stream: bool = False,
    ) -> dict[str, object]:
        predict_kwargs: dict[str, object] = {
            "source": source,
            "verbose": False,
        }
        if stream:
            predict_kwargs["stream"] = True
        if conf is not None:
            predict_kwargs["conf"] = conf
        if iou is not None:
            predict_kwargs["iou"] = iou
        return predict_kwargs

    def _read_video_metadata(self, video_path: Path) -> tuple[int | None, float | None]:
        if cv2 is None:
            return None, None
        capture = cv2.VideoCapture(str(video_path))
        try:
            if hasattr(capture, "isOpened") and not capture.isOpened():
                return None, None
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            return (total_frames or None), (fps or None)
        finally:
            capture.release()

    def _build_frame_result(self, result: object) -> CameraFrameResult:
        detections = self._build_detections(result)
        counts_by_label = dict(Counter(detection.label for detection in detections))
        return CameraFrameResult(
            detections=detections,
            counts_by_label=counts_by_label,
            total_detections=len(detections),
            annotated_frame=result.plot(),
        )

    def detect_image(
        self,
        image_path: str | Path,
        conf: float | None = None,
        iou: float | None = None,
    ) -> DetectionResult:
        image_path = Path(image_path)
        predict_kwargs = self._build_predict_kwargs(str(image_path), conf=conf, iou=iou)
        predictions = self._model.predict(**predict_kwargs)
        if not predictions:
            return DetectionResult(
                image_path=image_path,
                detections=[],
                counts_by_label={},
                total_detections=0,
                annotated_image=None,
            )

        frame_result = self._build_frame_result(predictions[0])
        return DetectionResult(
            image_path=image_path,
            detections=frame_result.detections,
            counts_by_label=frame_result.counts_by_label,
            total_detections=frame_result.total_detections,
            annotated_image=frame_result.annotated_frame,
        )

    def detect_images(
        self,
        image_paths: list[str | Path],
        conf: float | None = None,
        iou: float | None = None,
    ) -> list[DetectionResult]:
        return [self.detect_image(image_path, conf=conf, iou=iou) for image_path in image_paths]

    def detect_video(
        self,
        video_path: str | Path,
        conf: float | None = None,
        iou: float | None = None,
        progress_callback: Callable[[int, int | None], None] | None = None,
        frame_callback: Callable[[VideoFrameResult, int | None], None] | None = None,
    ) -> VideoDetectionResult:
        video_path = Path(video_path)
        total_frames, fps = self._read_video_metadata(video_path)
        predict_kwargs = self._build_predict_kwargs(str(video_path), conf=conf, iou=iou, stream=True)
        frame_results: list[VideoFrameResult] = []
        label_counter: Counter[str] = Counter()

        for frame_index, result in enumerate(self._model.predict(**predict_kwargs)):
            camera_frame_result = self._build_frame_result(result)
            label_counter.update(camera_frame_result.counts_by_label)
            timestamp_seconds = (frame_index / fps) if fps else 0.0
            frame_result = VideoFrameResult(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                detections=camera_frame_result.detections,
                counts_by_label=camera_frame_result.counts_by_label,
                total_detections=camera_frame_result.total_detections,
                annotated_frame=camera_frame_result.annotated_frame,
                raw_frame=getattr(result, "orig_img", None),
            )
            frame_results.append(frame_result)
            if progress_callback is not None:
                progress_callback(frame_index + 1, total_frames)
            if frame_callback is not None:
                frame_callback(frame_result, total_frames)

        return VideoDetectionResult(
            video_path=video_path,
            frames=frame_results,
            counts_by_label=dict(label_counter),
            total_detections=sum(frame.total_detections for frame in frame_results),
            total_frames=total_frames,
            processed_frames=len(frame_results),
            fps=fps,
        )

    def detect_frame(
        self,
        frame: object,
        conf: float | None = None,
        iou: float | None = None,
    ) -> CameraFrameResult:
        predict_kwargs = self._build_predict_kwargs(frame, conf=conf, iou=iou)
        predictions = self._model.predict(**predict_kwargs)
        if not predictions:
            return CameraFrameResult(
                detections=[],
                counts_by_label={},
                total_detections=0,
                annotated_frame=None,
            )
        return self._build_frame_result(predictions[0])
