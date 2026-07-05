@echo off
setlocal EnableDelayedExpansion

chcp 65001 >nul
for %%I in ("%~dp0.") do set "ROOT=%%~fI"
set "ENV=D:\Miniconda3\envs\yolo"
set "YOLO=%ENV%\Scripts\yolo.exe"
set "MODEL=%ROOT%\Reference\yolo26n.pt"
set "DATA=%ROOT%\lychee dataset\combined-detect.yaml"
set "PROJECT=%ROOT%\runs\lychee"
set "NAME=yolo26_merged_detect_safe"

if not exist "%ENV%\python.exe" (
    echo [ERROR] Conda environment not found: !ENV!
    pause
    exit /b 1
)

if not exist "!YOLO!" (
    echo [ERROR] YOLO launcher not found: !YOLO!
    pause
    exit /b 1
)

if not exist "!MODEL!" (
    echo [ERROR] Base model not found: !MODEL!
    pause
    exit /b 1
)

if not exist "!DATA!" (
    echo [ERROR] Dataset config not found: !DATA!
    pause
    exit /b 1
)

echo [INFO] Training detection model with merged lychee datasets
echo [INFO] Model   : !MODEL!
echo [INFO] Data    : !DATA!
echo [INFO] Project : !PROJECT!
echo [INFO] Name    : !NAME!
echo [INFO] Settings: epochs=80 imgsz=448 batch=1 workers=0 amp=False augment=False mosaic=0.0
echo.

if not exist "!PROJECT!" mkdir "!PROJECT!"

"!YOLO!" detect train ^
 model="!MODEL!" ^
 data="!DATA!" ^
 epochs=80 ^
 imgsz=448 ^
 batch=1 ^
 device=0 ^
 amp=False ^
 workers=0 ^
 patience=30 ^
 pretrained=True ^
 cache=False ^
 augment=False ^
 mosaic=0.0 ^
 mixup=0.0 ^
 copy_paste=0.0 ^
 close_mosaic=0 ^
 plots=False ^
 save_period=-1 ^
 exist_ok=True ^
 project="!PROJECT!" ^
 name="!NAME!"

if errorlevel 1 (
    echo [ERROR] Training failed.
    pause
    exit /b 1
)

echo.
echo [INFO] Training finished.
echo [INFO] Best weights: !PROJECT!\!NAME!\weights\best.pt
pause
endlocal
