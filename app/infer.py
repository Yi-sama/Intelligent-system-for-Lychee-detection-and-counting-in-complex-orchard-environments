from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ultralytics import YOLO


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


class DetectionService:
    def __init__(self, model_path: str | Path, model_loader: Callable[[str | Path], object] | None = None):
        self.model_path = Path(model_path)
        self._model_loader = model_loader or YOLO
        self._model = self._model_loader(str(self.model_path))

    def detect_image(self, image_path: str | Path) -> DetectionResult:
        image_path = Path(image_path)
        predictions = self._model.predict(source=str(image_path), verbose=False)
        if not predictions:
            return DetectionResult(
                image_path=image_path,
                detections=[],
                counts_by_label={},
                total_detections=0,
                annotated_image=None,
            )

        result = predictions[0]
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

        counts_by_label = dict(Counter(detection.label for detection in detections))
        return DetectionResult(
            image_path=image_path,
            detections=detections,
            counts_by_label=counts_by_label,
            total_detections=len(detections),
            annotated_image=result.plot(),
        )
