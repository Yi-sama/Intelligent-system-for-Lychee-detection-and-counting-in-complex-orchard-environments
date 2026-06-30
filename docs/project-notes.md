# Project Notes

## Current Status

- Objective: Build a desktop prototype for lychee-related object detection/counting, starting with the image workflow only.
- Current state: A minimal PyQt5 image prototype now exists with local YOLO inference, count summary, CSV export, and annotated-image export.
- Next step: Finalize the next-iteration PyQt5 GUI design around the image workflow, then refactor the UI accordingly.
- Blockers: No project-specific lychee-trained weight has been confirmed yet; current smoke test on `Reference/yolo26n.pt` returned `apple` labels, so the pipeline works but task accuracy is not yet aligned.

## Environment

- Preferred environment: `D:\Miniconda3\envs\yolo`
- Confirmed imports on 2026-06-30: `PyQt5`, `ultralytics 8.4.63`
- Verification can rely on `unittest`; do not assume `pytest` is installed in the current `yolo` environment.

## Commands

- Run unit tests:
  - `conda run -p D:\Miniconda3\envs\yolo python -m unittest tests.test_app`
- Launch the desktop app:
  - `conda run -p D:\Miniconda3\envs\yolo python app/main.py`
- Real inference smoke test pattern:
  - `conda run -p D:\Miniconda3\envs\yolo python -c "... DetectionService(...).detect_image(...)"` with `PYTHONPATH` set to the project root

## Important Data

- Current local model weight: `Reference/yolo26n.pt`
- Current local image/video samples: `Reference/TestFiles/`
- Smoke-test outputs are written to: `outputs/`

## Decisions

- Scope is intentionally constrained to the image workflow first: select image, run detection, show results, export CSV, export annotated image.
- Tests use standard-library `unittest` instead of `pytest` because the current `yolo` environment does not expose `pytest`.
- GUI evolution should follow a workbench-style layout instead of splitting the first polished version into separate mode pages too early.
- Video and camera should be preserved as future expansion points in the UI, but should not complicate the first cleaned-up image workflow screen.

## GUI Design Direction

- Keep `PyQt5` as the GUI framework and keep `app/main.py` as the desktop entry point.
- Preserve the existing separation where GUI calls a detection service instead of embedding YOLO logic directly into the window layer.
- Preferred layout for the next iteration:
  - left control column for input mode and actions
  - center preview panel for the current image/result
  - right result panel for counts, detail list, and lightweight run/export status
- Mandatory first-class features in the UI:
  - choose image
  - run detection
  - show annotated result
  - show total count and per-label count
  - export CSV
  - export result image
- Known current GUI gaps to address in the redesign:
  - only image mode is usable
  - synchronous detection may freeze the window
  - no running/progress state
  - limited export control
  - visible text/encoding problems in the current Chinese UI strings

## Durable Debug Findings

- `conda run` currently prints OpenCL vendor/temp-file warnings:
  - `The system cannot find the file specified.`
  - `Could Not Find D:\Miniconda3\envs\yolo\Library\etc\OpenCL\vendors\temp.txt`
  These warnings did not block `PyQt5` import, `ultralytics` import, unit tests, or real image inference on 2026-06-30.

## Journal Note 09:47

- 2026-06-30: 已完成任务书与仓库现状梳理。当前仓库只有任务书与 Reference 参考资源，不是完整自研工程。用户说明已有 Anaconda YOLO 环境，待确认环境名和可用依赖。任务书明确包含图片/视频/摄像头三类推理、结果保存与 CSV 导出、桌面 GUI；数据集不少于 2163 张是文档要求，但是否必须自行采集仍需结合课程口径和现有数据判断。

## Journal Note 09:49

- 2026-06-30: 本机 Conda 可见环境为 base、ancoda_getspark、blender_rag、yolo。项目可复用的主要环境为 D:\Miniconda3\envs\yolo，其中已确认 Python 3.11.15、PyTorch 2.5.1、CUDA 可用、ultralytics 8.4.63、tkinter 可用，当前未发现 PyQt5。任务书已核实：图片/视频/摄像头三类推理、结果保存与 CSV 导出、总数统计、桌面 GUI 均为明确要求；GUI 框架允许 PyQt5 或 tkinter，但文档整体主推 PyQt5；数据集不少于 2163 张为硬性下限。

## Journal Note 09:49

- 2026-06-30: 本机 Conda 可见环境为 base、ancoda_getspark、blender_rag、yolo。项目可复用的主要环境为 D:\Miniconda3\envs\yolo，其中已确认 Python 3.11.15、PyTorch 2.5.1、CUDA 可用、ultralytics 8.4.63、tkinter 可用，当前未发现 PyQt5。任务书已核实：图片/视频/摄像头三类推理、结果保存与 CSV 导出、总数统计、桌面 GUI 均为明确要求；GUI 框架允许 PyQt5 或 tkinter，但文档整体主推 PyQt5；数据集不少于 2163 张为硬性下限。

## Journal Note 10:17

- 2026-06-30: 用户决定 GUI 采用 PyQt5。已核对外部数据集候选 SeiriosLab/Lychee（GitHub README）：仓库说明真实数据迁移到 Mendeley 数据页，README 描述数据集包含 11,414 张高分辨率图像，检测标注格式支持 YOLO v5/v8 txt，目录中区分 detection/raw、split(train/val/test)、augmentation、depth_estimate。该数据集可作为满足课程数据量要求的候选来源，但项目当前优先应取 detection 任务相关 RGB 图像与 YOLO 检测标注，不必一开始引入成熟度分类、抓取标注或深度图。

## Journal Note 10:30

- 2026-06-30: 用户澄清 docx 仅作为参考，不作为必须完全对齐的硬性验收标准。项目数据集当前直接使用工作区现有目录 D:\ALL OF STUDY\实训作业\智能应用及其算法\Instruction(1)\Reference\TestFiles；虽然样本量不大，但优先基于该目录完成系统搭建与功能打通，不再以外部 Lychee 数据集作为当前实施前提。

## Journal Note 10:40

- 2026-06-30: 用户明确要求后续工作按 project-progress-journal 维护项目记录。之后需持续把已完成进展写入 docs/progress，当期有效的环境/决策/路径信息写入 docs/project-notes，调试过程写入 docs/debug。

## Journal Note 10:40

- 2026-06-30: 用户明确要求后续工作按 project-progress-journal 维护项目记录。之后需持续把已完成进展写入 docs/progress，当期有效的环境/决策/路径信息写入 docs/project-notes，调试过程写入 docs/debug。
