# Intelligent System for Lychee Detection and Counting in Complex Orchard Environments

PyQt5 desktop application for lychee detection, counting, and result export in complex orchard scenarios. The project integrates image inference, batch image processing, local video detection, camera preview/detection, and training script support in one Windows-oriented workflow.

## Project Status

This repository reflects a completed course-project style delivery with a runnable desktop prototype and supporting training workflow.

Implemented scope:

- Single-image detection
- Batch image detection
- Local video frame-by-frame detection
- Local camera preview and realtime detection baseline
- CSV export
- Annotated image export
- Annotated video export
- YOLO training launcher
- Unit-test coverage for inference, export, GUI layout, batch flow, video flow, and camera flow

## Main Features

- Unified PyQt5 desktop interface with image, video, and camera modes
- Adjustable confidence and IoU thresholds
- Preview-first interaction design
- Result summary plus per-target detail display
- Batch image folder workflow
- Video progress updates and annotated video replay
- Camera preview with device release handling
- Export of structured CSV result data

## Project Structure

```text
app/                    Desktop GUI, inference service, export logic
tests/                  Automated tests
docs/                   Progress notes, debug notes, project notes
Reference/              Base model, dependencies, reference resources
outputs/                Inference output directory
lychee dataset/         Local training/inference dataset directory (ignored by Git)
start.bat               Desktop launcher
train.bat               Training launcher
```

## Environment

- OS: Windows
- Recommended Python environment: `D:\Miniconda3\envs\yolo`
- Core dependencies:
  - `PyQt5`
  - `ultralytics`
  - `opencv-python`
  - `numpy`
  - `Pillow`
  - `torch`

Install package requirements from:

```powershell
pip install -r Reference\requirements.txt
```

## Quick Start

1. Prepare the Conda environment and dependencies.
2. Make sure the default model exists at `Reference\yolo26n.pt`.
3. Launch the desktop app:

```powershell
start.bat
```

Alternative entry:

```powershell
conda run -p D:\Miniconda3\envs\yolo python -m app.main
```

## Training

Training is launched separately from the GUI:

```powershell
train.bat
```

Current training configuration:

- Base model: `Reference\yolo26n.pt`
- Dataset config: `lychee dataset\combined-detect.yaml`
- Output directory: `runs\lychee\yolo26_merged_detect_safe`

Expected trained weight:

```text
runs\lychee\yolo26_merged_detect_safe\weights\best.pt
```

After training, the trained weight can replace the generic default model for project-specific lychee inference.

## Testing

Run tests with the intended environment:

```powershell
conda run -p D:\Miniconda3\envs\yolo python -m unittest tests.test_app
```

For lightweight syntax verification:

```powershell
python -m py_compile app\gui.py app\infer.py app\export.py tests\test_app.py
```

## Notes

- The repository ignores local datasets, training runs, generated outputs, and process artifacts.
- The current workflow is optimized for the local Windows + Conda environment above.
- Default launcher paths are environment-specific and may need adjustment on another machine.

