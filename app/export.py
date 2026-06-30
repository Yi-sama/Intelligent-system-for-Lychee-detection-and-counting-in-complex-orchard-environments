from __future__ import annotations

import csv
from pathlib import Path

import cv2

from app.infer import DetectionResult


def export_detections_csv(result: DetectionResult, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "label", "confidence", "x1", "y1", "x2", "y2"],
        )
        writer.writeheader()
        for detection in result.detections:
            writer.writerow(
                {
                    "image_path": str(result.image_path),
                    "label": detection.label,
                    "confidence": f"{detection.confidence:.4f}",
                    "x1": str(detection.bbox[0]),
                    "y1": str(detection.bbox[1]),
                    "x2": str(detection.bbox[2]),
                    "y2": str(detection.bbox[3]),
                }
            )

    return output_path


def export_annotated_image(result: DetectionResult, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_data = result.annotated_image
    if image_data is None:
        raise ValueError("No annotated image available to export.")

    if isinstance(image_data, bytes):
        output_path.write_bytes(image_data)
        return output_path

    if hasattr(image_data, "tofile"):
        success = cv2.imwrite(str(output_path), image_data)
        if not success:
            raise ValueError(f"Failed to write image to {output_path}")
        return output_path

    raise TypeError(f"Unsupported annotated image type: {type(image_data)!r}")
