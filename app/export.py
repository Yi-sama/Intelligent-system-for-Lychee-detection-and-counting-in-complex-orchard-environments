from __future__ import annotations

import csv
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - depends on local env
    cv2 = None

from app.infer import DetectionResult, VideoDetectionResult


def _write_detection_rows(writer: csv.DictWriter, results: list[DetectionResult]) -> None:
    for result in results:
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


def export_detections_csv(result: DetectionResult, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "label", "confidence", "x1", "y1", "x2", "y2"],
        )
        writer.writeheader()
        _write_detection_rows(writer, [result])

    return output_path


def export_detections_csv_batch(results: list[DetectionResult], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "label", "confidence", "x1", "y1", "x2", "y2"],
        )
        writer.writeheader()
        _write_detection_rows(writer, results)

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


def export_annotated_images(results: list[DetectionResult], output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_paths: list[Path] = []
    for result in results:
        suffix = result.image_path.suffix or ".png"
        output_path = output_dir / f"{result.image_path.stem}_annotated{suffix}"
        exported_paths.append(export_annotated_image(result, output_path))
    return exported_paths


def export_video_detections_csv(result: VideoDetectionResult, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video_path",
                "frame_index",
                "timestamp_seconds",
                "label",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
            ],
        )
        writer.writeheader()
        for frame in result.frames:
            for detection in frame.detections:
                writer.writerow(
                    {
                        "video_path": str(result.video_path),
                        "frame_index": str(frame.frame_index),
                        "timestamp_seconds": f"{frame.timestamp_seconds:.4f}",
                        "label": detection.label,
                        "confidence": f"{detection.confidence:.4f}",
                        "x1": str(detection.bbox[0]),
                        "y1": str(detection.bbox[1]),
                        "x2": str(detection.bbox[2]),
                        "y2": str(detection.bbox[3]),
                    }
                )

    return output_path


def _frame_size(frame: object) -> tuple[int, int]:
    if hasattr(frame, "shape"):
        height, width = frame.shape[:2]
        return int(width), int(height)
    if isinstance(frame, (list, tuple)) and frame:
        height = len(frame)
        width = len(frame[0]) if frame[0] else 0
        return width, height
    raise TypeError(f"Unsupported annotated frame type: {type(frame)!r}")


def export_annotated_video(result: VideoDetectionResult, output_path: str | Path) -> Path:
    if cv2 is None:
        raise ModuleNotFoundError("cv2 is required to export annotated videos.")
    if not result.frames:
        raise ValueError("No annotated video frames available to export.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_frame = result.frames[0].annotated_frame
    frame_size = _frame_size(first_frame)
    fps = result.fps or 30.0
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    try:
        for frame in result.frames:
            writer.write(frame.annotated_frame)
    finally:
        writer.release()

    return output_path
