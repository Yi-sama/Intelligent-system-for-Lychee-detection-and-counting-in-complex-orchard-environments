# Project Notes

## Current Status

- Objective: Maintain the PyQt5 desktop prototype for lychee-related detection/counting across image, local video, and camera workflows.
- Current state: The three-mode desktop shell (`图片 / 视频 / 摄像头`) is implemented with a shared preview-first layout, an end-to-end image workflow, a completed local video MVP, and a camera preview baseline.
- Next step: Verify the video and camera GUI paths inside the intended `D:\Miniconda3\envs\yolo` runtime, then continue from the current MVP toward fuller camera detection behavior and project-specific model validation.
- Blockers: The active interpreter used for local command-line verification does not provide `PyQt5`, so GUI tests are skipped there. No project-specific lychee-trained weight has been confirmed yet; current smoke testing on `Reference/yolo26n.pt` still returns `apple` labels.

## Environment

- Preferred environment: `D:\Miniconda3\envs\yolo`
- Confirmed imports on 2026-06-30: `PyQt5`, `ultralytics 8.4.63`
- Verification can rely on `unittest`; do not assume `pytest` is installed in the current `yolo` environment.
- Primary UI tuning screen during current preview-layout work: `2560 x 1600`

## Commands

- Run unit tests:
  - `conda run -p D:\Miniconda3\envs\yolo python -m unittest tests.test_app`
- Launch the desktop app:
  - `conda run -p D:\Miniconda3\envs\yolo python app/main.py`
  - `start.bat`
- Real inference smoke test pattern:
  - `conda run -p D:\Miniconda3\envs\yolo python -c "... DetectionService(...).detect_image(...)"` with `PYTHONPATH` set to the project root

## Important Data

- Current local model weight: `Reference/yolo26n.pt`
- Current local image/video samples: `Reference/TestFiles/`
- Smoke-test outputs are written to: `outputs/`

## Decisions

- The initial delivery sequence was image first, then local video, then camera; the local video MVP is now complete and camera remains the next functional expansion area.
- Tests use standard-library `unittest` instead of `pytest` because the current `yolo` environment does not expose `pytest`.
- GUI evolution should prioritize a preview-first lightweight desktop layout instead of a dense workbench or early split into separate mode pages.
- Image, video, and camera should continue sharing one PyQt5 shell; new work should extend the existing mode switch, preview, result, and export patterns instead of introducing separate windows.
- The main preview stage minimum size now uses a percentage rule instead of a fixed pixel constant: `75%` of the active screen's shorter side, with a `900px` floor. On the current `2560 x 1600` screen, this resolves to `1200 x 1200`.
- Preview-stage tuning should keep the large stage and adjust image alignment independently; reducing stage size is not a good substitute for reducing perceived letterboxing.

## Video Mode Requirements And GUI Integration

- Source document reviewed: `复杂果园环境下荔枝检测与计数智能系统.docx`.
- The document explicitly requires video mode, not just a reserved UI entry:
  - local video逐帧检测;
  - real-time count updates during video processing;
  - output of an annotated result video with detection boxes;
  - adjustable confidence and IoU thresholds;
  - one-click integration with the GUI buttons;
  - result display including total count, elapsed time, per-target confidence, and coordinates;
  - export support for image/video results and CSV data.
- The document also requires camera mode: local camera input with real-time detection and dynamic counting. Treat camera as the next stage after local video because video files are easier to implement and verify against the current GUI.
- The document only constrains the GUI at a functional-area level: file selection, parameter settings, image/video display, and result information. It does not mandate a specific layout, so the existing PyQt5 layout should be reused.
- Current GUI already has the correct integration skeleton:
  - top-level `image / video / camera` mode buttons in `app/gui.py`;
  - `QStackedWidget` mode-specific configuration area;
  - left-side configuration and result panels;
  - right-side dominant preview stage;
  - `run_detection()` as the central action entry point.
- First video-mode MVP should use the existing GUI structure instead of creating a separate window:
  - replace the `video-config-panel` placeholder with a real video configuration panel;
  - add local video selection with common formats such as `mp4 / avi / mov / mkv`;
  - reuse `conf`, `iou`, label visibility, and export-format controls where possible;
  - show the current processed frame in the existing preview area;
  - add simple progress/frame status rather than a full media-player timeline in the first version;
  - show current-frame count, total/processed-frame summary, elapsed time, and per-frame detection details in the existing result area;
  - export an annotated video and, if practical, a frame-level CSV.
- Do not force video into the existing image-state model. Add dedicated video state such as `current_video_path`, current frame index, frame count/FPS metadata, and a video result collection.
- `DetectionService` should grow a dedicated video API rather than overloading `detect_image()`:
  - likely shape: `detect_video(video_path, conf, iou, show_labels, progress_callback)`;
  - the implementation can be based on OpenCV frame reading plus YOLO prediction, matching the current service-layer pattern.
- Main technical risk: synchronous video processing will freeze the PyQt window. Video detection should eventually move to a worker thread or use progress callbacks, even if the first implementation is minimal.
- Recommended scope order:
  - first: local video file mode with逐帧检测, live frame preview, count updates, and annotated-video export;
  - second: frame-level CSV export and better progress controls;
  - third: camera mode with real-time capture, device release, and dynamic counting;
  - later only if needed: tracking IDs, full timeline/player controls, and multi-device camera selection.

## GUI Design Direction

- Keep `PyQt5` as the GUI framework and keep `app/main.py` as the desktop entry point.
- Preserve the existing separation where GUI calls a detection service instead of embedding YOLO logic directly into the window layer.
- Preferred layout for the next iteration:
  - a top mode-switch row reserved for exactly three large mode blocks: `图片`, `视频`, `摄像头`
  - a thin status row below the mode blocks for model name, current file name, readiness/running state, and elapsed time
  - a narrow left-side configuration area for mode-specific options such as `conf`, `iou`, label visibility, export format, and source/device selectors
  - a large preview area on the right that remains the visual center of the screen
  - a compact preview-local action strip near the preview for `打开`, `开始检测`, `原图/结果切换`, and `导出`
  - a bottom information strip that shows both count summary and detection details at the same time
- Visual priority:
  - the preview area should remain the dominant region of the window, roughly 60%-75% of the perceived layout weight
  - the screen should feel sparse and direct rather than panel-heavy
- Top-area discipline:
  - the top row should express mode switching only and must not be turned into a parameter toolbar
  - mode switching should read as three clear entry blocks rather than a dense tab strip full of auxiliary controls
- Left-side discipline:
  - the left area is a configuration zone, not the primary action zone and not a decorative empty column
  - keep the column narrow and information-dense, with short labels and mode-specific controls only
- Action-area conventions:
  - eye icon = open/select image
  - run/play icon = start detection
  - export icon = open an export chooser for `CSV` or result image
  - avoid tall button columns and avoid exposing too many actions at once
- Preview action conventions:
  - preview-related actions should stay attached to the preview area instead of being moved to the global header
  - prefer icon-first buttons with short labels instead of long descriptive buttons or helper headings
- Interaction conventions:
  - clicking the preview image should toggle original image and annotated result image
  - the dedicated `预览` button should open a larger preview dialog
  - mouse wheel should zoom the image inside the preview dialog directly
  - summary and detail views should remain visible together instead of being hidden behind tabs
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
- Scope discipline for this redesign:
  - fully clean up the image workflow first
  - keep `视频` and `摄像头` visible as first-class mode entry points in the header
  - reveal their controls in the left configuration area only when that mode is active, rather than crowding the default image screen
  - treat `视频` and `摄像头` as layout-complete placeholders until their detection pipeline is implemented; the current complete workflow remains `图片`
  - keep the image-mode bottom area lightweight: a short `count summary + detection details` result band is sufficient; a tall third `目标列表` block is not required by the current document constraints
  - keep the batch-image selector visible as a reserved image-mode option, but do not present it as a complete workflow until batch behavior is genuinely implemented

## Video Mode Requirements (2026-07-03)

- Source reviewed: `复杂果园环境下荔枝检测与计数智能系统.docx`.
- The document explicitly requires video mode as a real feature, not just a reserved GUI entry.
- Required video-mode capabilities confirmed from the document:
  - local video frame-by-frame detection;
  - real-time count updates while processing the video;
  - annotated result-video export with detection boxes;
  - adjustable `conf` and `IoU` thresholds;
  - GUI binding so image, video, and camera modes can all be started from the desktop app;
  - result display for total count, elapsed time, confidence, and coordinate details;
  - export support for image/video outputs and CSV data.
- The document also requires camera mode with real-time local-camera detection and dynamic counting, but local video should be implemented first because it is lower risk and closer to the current image workflow.
- The document does not prescribe an exact layout. It only requires these functional regions:
  - file selection;
  - parameter settings;
  - image/video display;
  - result information.

## Video Mode GUI Integration (2026-07-03)

- The current PyQt5 GUI already has the correct high-level skeleton for video mode:
  - top-level `image / video / camera` mode entries;
  - a mode-specific `QStackedWidget` config area;
  - a left configuration/result region;
  - a dominant right-side preview region;
  - `run_detection()` as the main workflow entry.
- Recommended integration path:
  - keep the existing `MainWindow` and extend it;
  - replace the `video-config-panel` placeholder with a real video configuration panel;
  - route video mode through `mode-entry-video -> _apply_mode("video") -> video-config-panel -> run_detection()`.
- First video-mode MVP should stay aligned with the current GUI structure:
  - select a local video file;
  - reuse `conf`, `IoU`, label visibility, and export settings where possible;
  - show the current processed frame in the existing preview area;
  - display count summary, elapsed time, and frame-level detection details in the existing result area;
  - export an annotated result video;
  - add simple frame/progress feedback rather than a full timeline player in the first version.
- Do not force video through the current image-only state model. Video mode needs dedicated state such as:
  - `current_video_path`;
  - current frame index;
  - frame-count/FPS metadata;
  - a video-result collection.
- The service layer should gain a dedicated video API rather than overloading `detect_image()`.
- Main technical risk: synchronous video inference will freeze the PyQt GUI. Video mode will likely need a worker thread or at least progress-callback-based separation from the main UI loop.
- Recommended delivery order:
  - first: local video mode;
  - second: richer progress/export details;
  - third: camera mode;
  - later only if needed: tracking IDs, full timeline controls, and multi-device camera selection.

## Video Mode Requirements (2026-07-03)

- Source reviewed: `复杂果园环境下荔枝检测与计数智能系统.docx`.
- The document explicitly requires video mode as a real feature, not just a reserved GUI entry.
- Required video-mode capabilities confirmed from the document:
  - local video frame-by-frame detection;
  - real-time count updates while processing the video;
  - annotated result-video export with detection boxes;
  - adjustable `conf` and `IoU` thresholds;
  - GUI binding so image, video, and camera modes can all be started from the desktop app;
  - result display for total count, elapsed time, confidence, and coordinate details;
  - export support for image/video outputs and CSV data.
- The document also requires camera mode with real-time local-camera detection and dynamic counting, but local video should be implemented first because it is lower risk and closer to the current image workflow.
- The document does not prescribe an exact layout. It only requires these functional regions:
  - file selection;
  - parameter settings;
  - image/video display;
  - result information.

## Video Mode GUI Integration (2026-07-03)

- The current PyQt5 GUI already has the correct high-level skeleton for video mode:
  - top-level `image / video / camera` mode entries;
  - a mode-specific `QStackedWidget` config area;
  - a left configuration/result region;
  - a dominant right-side preview region;
  - `run_detection()` as the main workflow entry.
- Recommended integration path:
  - keep the existing `MainWindow` and extend it;
  - replace the `video-config-panel` placeholder with a real video configuration panel;
  - route video mode through `mode-entry-video -> _apply_mode("video") -> video-config-panel -> run_detection()`.
- First video-mode MVP should stay aligned with the current GUI structure:
  - select a local video file;
  - reuse `conf`, `IoU`, label visibility, and export settings where possible;
  - show the current processed frame in the existing preview area;
  - display count summary, elapsed time, and frame-level detection details in the existing result area;
  - export an annotated result video;
  - add simple frame/progress feedback rather than a full timeline player in the first version.
- Do not force video through the current image-only state model. Video mode needs dedicated state such as:
  - `current_video_path`;
  - current frame index;
  - frame-count/FPS metadata;
  - a video-result collection.
- The service layer should gain a dedicated video API rather than overloading `detect_image()`.
- Main technical risk: synchronous video inference will freeze the PyQt GUI. Video mode will likely need a worker thread or at least progress-callback-based separation from the main UI loop.
- Recommended delivery order:
  - first: local video mode;
  - second: richer progress/export details;
  - third: camera mode;
  - later only if needed: tracking IDs, full timeline controls, and multi-device camera selection.

## Durable Debug Findings

- `conda run` currently prints OpenCL vendor/temp-file warnings:
  - `The system cannot find the file specified.`
  - `Could Not Find D:\Miniconda3\envs\yolo\Library\etc\OpenCL\vendors\temp.txt`
  These warnings did not block `PyQt5` import, `ultralytics` import, unit tests, or real image inference on 2026-06-30.
- 2026-07-03 verification boundary:
  - Local non-GUI verification passed in the active interpreter for the new video service/export path.
  - GUI-targeted `unittest` coverage currently reports `OK (skipped=5)` in the active interpreter because `PyQt5` is missing there.
  - Treat the current local evidence for video mode as service/export tests plus `py_compile`; true GUI runtime verification still belongs in `D:\Miniconda3\envs\yolo`.
- 2026-07-03 durable update:
  - Image mode now includes a folder-based batch-image workflow under `图片来源 -> 文件夹批量`, reusing the same left-side `conf / iou / export` controls as single-image mode.
  - Batch mode currently scans only the selected folder's top-level `png / jpg / jpeg / bmp` files and exports one merged CSV plus one annotated-image directory.
  - GUI source switching and mode switching now preserve any image state that the user has not explicitly closed.
  - Single-image close now means true removal: after clicking `×`, the image/result state is cleared and will not reappear after switching to `视频`/`摄像头` and back.
  - Batch-image close now means true removal from the batch list: closing the current image removes it from `batch_image_paths` and `batch_results`, then immediately advances to the next valid image or the new last image.
  - Batch-image `全部关闭` now clears the entire batch state rather than only hiding the current display.
  - The lightweight executable verification command for this GUI behavior update is `python -m py_compile app\gui.py tests\test_app.py`.
  - 2026-07-03 video-mode durable update:
    - `app/infer.py` now exposes `VideoFrameResult`, `VideoDetectionResult`, and a dedicated `DetectionService.detect_video(...)`.
    - `detect_video(...)` now supports both `progress_callback(processed, total)` and `frame_callback(frame_result, total)` so the GUI can update frame progress without embedding YOLO internals in the window layer.
    - `VideoFrameResult` now also carries an optional `raw_frame` payload so the GUI can show the paused video's current original frame instead of falling back to a single static preview image.
    - `app/export.py` now supports `export_video_detections_csv(...)` and `export_annotated_video(...)`.
    - `app/gui.py` now includes a real video config panel, dedicated video-mode state, a local video picker, and a `QThread` worker path so local video inference no longer runs on the main GUI thread.
    - The current video MVP reuses the existing preview/result areas for selected-video preview, processed-frame updates, aggregated totals, per-frame detail lines, CSV export, and annotated-video export.
    - Video-result playback now uses a lightweight GUI-side timer rather than a media-player widget: once detection finishes, the annotated frames auto-play in the main preview area; clicking the preview toggles pause/resume; `原图` / `结果图` become explicit video-only buttons and stay disabled until paused; `重新开始` replays from frame 0.
    - 2026-07-03 follow-up interaction correction: video preview clicks should never drop into the generic “please detect first” image-style prompt when a video has already been opened; video mode now routes preview clicks through the video pause/resume state first, and the paused-frame original/result switch is exposed as one dynamic button whose label shows the target state rather than two fixed buttons.
    - In the active interpreter, GUI runtime verification is still blocked because `PyQt5` is missing; local evidence currently comes from passing non-GUI `unittest` cases and `python -m py_compile`.

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
## Journal Note 2026-07-03 Camera And Environment

- Camera mode no longer stops at a placeholder-only state machine: `app/gui.py` now has a real camera config panel and a local-camera preview baseline using `cv2.VideoCapture + QTimer`.
- The root cause of “打开摄像头后没有画面” was an implementation gap rather than a missing package: the earlier camera MVP changed only labels and button state, but did not open a capture device or render frames into the preview area.
- Camera preview now releases the capture device when closing the camera, leaving camera mode, or closing the window.
- The intended runtime environment remains `D:\Miniconda3\envs\yolo`, and revalidation on 2026-07-03 confirmed that `PyQt5`, `cv2`, `ultralytics`, and `numpy` are all importable there.
- The local `base` interpreter still lacks `PyQt5`/`cv2`, so direct `python ...` verification from `base` should not be used to judge project readiness; use `conda run -p D:\Miniconda3\envs\yolo ...` or `start.bat`.

## Journal Note 2026-07-03 Lychee Training

- Training for the lychee model does not need to happen inside the current GUI. The current desktop app is an inference/display/export shell; training is being handled externally through `train.bat`.
- User training dataset root: `D:\ALL OF STUDY\实训作业\智能应用及其算法\Instruction(1)\lychee dataset`.
- Merged detect-training config: `lychee dataset\combined-detect.yaml`.
- Current merged config uses both datasets for training:
  - `lychee2/train/images`
  - `lychee3/train/images`
  - validation/test currently come from `lychee2`
- `lychee2` is a detection dataset, while `lychee3` contributes segmentation-style labels. For the current detect-model goal, Ultralytics can still train by dropping segment masks and using boxes.
- Current training launcher: `train.bat`.
- `train.bat` should call `D:\Miniconda3\envs\yolo\Scripts\yolo.exe` directly instead of `conda run ...` on this machine, because `conda run` is prone to encoding/plugin-output failures in the Chinese-path environment.
- Current overnight training profile in `train.bat` is intentionally conservative for stability:
  - `epochs=80`
  - `imgsz=448`
  - `batch=1`
  - `workers=0`
  - `amp=False`
  - `augment=False`
  - `mosaic=0.0`
  - `mixup=0.0`
  - `copy_paste=0.0`
  - `close_mosaic=0`
- Expected trained weight output: `runs\lychee\yolo26_merged_detect_safe\weights\best.pt`.
- The generic `Reference\yolo26n.pt` explains why current smoke inference reports `apple`; once the lychee-trained `best.pt` is loaded, the class name should become `lychee`.
- Older run directories remain under `runs\lychee\`. When comparing logs, confirm the active run name and `args.yaml` before trusting pasted epoch/parameter output.
- Known current blocker for training stability:
  - repeated failures came from Windows-side CPU memory / virtual-memory pressure during image loading or augmentation, not only GPU VRAM.
