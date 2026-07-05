@echo off
setlocal

cd /d "%~dp0"

set "CONDA_ENV=D:\Miniconda3\envs\yolo"

where conda >nul 2>nul
if errorlevel 1 (
    echo [ERROR] conda command not found. Please make sure Miniconda/Anaconda is installed and added to PATH.
    pause
    exit /b 1
)

if not exist "%CONDA_ENV%\python.exe" (
    echo [ERROR] Conda environment not found: %CONDA_ENV%
    pause
    exit /b 1
)

echo Starting Lychee Detection Console...
conda run -p "%CONDA_ENV%" python -m app.main

if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with an error.
    pause
    exit /b 1
)

endlocal
