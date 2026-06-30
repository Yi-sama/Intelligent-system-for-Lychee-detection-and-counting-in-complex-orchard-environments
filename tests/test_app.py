import csv
import shutil
import unittest
from pathlib import Path

import numpy as np

from app.export import export_annotated_image, export_detections_csv
from app.infer import Detection, DetectionResult, DetectionService


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
    def predict(self, source, verbose=False):
        return [FakeResult()]


def fake_model_loader(_model_path):
    return FakeModel()


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


if __name__ == "__main__":
    unittest.main()
