@echo off
REM ============================================================================
REM Patch Virtual Environment Activation Script with CUDA Settings
REM ============================================================================
REM This script adds CUDA environment variables to .venv activation
REM Run once after creating/recreating the virtual environment
REM ============================================================================

echo Patching .venv activation script with CUDA settings...

set ACTIVATE_SCRIPT=.venv\Scripts\activate.bat

if not exist "%ACTIVATE_SCRIPT%" (
    echo ERROR: Virtual environment not found at .venv
    echo Please create the virtual environment first with: uv sync
    exit /b 1
)

REM Check if already patched
findstr /C:"CUDA 13.1 NVRTC fix" "%ACTIVATE_SCRIPT%" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Virtual environment already patched with CUDA settings.
    exit /b 0
)

REM Backup original't-
copy "%ACTIVATE_SCRIPT%" "%ACTIVATE_SCRIPT%.backup" >nul

REM Add CUDA settings to activation script
echo. >> "%ACTIVATE_SCRIPT%"
echo REM --- CUDA 13.1 NVRTC fix --- >> "%ACTIVATE_SCRIPT%"
echo set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1 >> "%ACTIVATE_SCRIPT%"
echo set PATH=%%CUDA_PATH%%\bin\x64;%%CUDA_PATH%%\nvvm\bin\x64;%%PATH%% >> "%ACTIVATE_SCRIPT%"
echo. >> "%ACTIVATE_SCRIPT%"
echo REM --- cuTile Environment Variables --- >> "%ACTIVATE_SCRIPT%"
echo set CUDA_TILE_ENABLE_CRASH_DUMP=1 >> "%ACTIVATE_SCRIPT%"
echo set CUDA_TILE_COMPILER_TIMEOUT_SEC=300 >> "%ACTIVATE_SCRIPT%"
echo set CUDA_TILE_LOGS=CUTILEIR >> "%ACTIVATE_SCRIPT%"
echo set CUDA_TILE_TEMP_DIR=%%TEMP%%\cuda_tile >> "%ACTIVATE_SCRIPT%"

echo.
echo SUCCESS! Virtual environment activation script patched.
echo.
echo The following environment variables will now be set automatically
echo when you activate the virtual environment:
echo   - CUDA_PATH
echo   - CUDA_TILE_ENABLE_CRASH_DUMP
echo   - CUDA_TILE_COMPILER_TIMEOUT_SEC
echo   - CUDA_TILE_LOGS
echo   - CUDA_TILE_TEMP_DIR
echo.
echo To activate: .venv\Scripts\activate.bat
echo.
