@echo off
REM ============================================================================
REM CUDA Environment Setup Script
REM ============================================================================
REM This script sets up CUDA paths and cuTile environment variables
REM Usage: Run this script before executing CUDA-dependent Python code
REM       scripts\setup_cuda_env.bat
REM ============================================================================

echo Setting up CUDA environment...

REM --- CUDA 13.1 NVRTC fix ---
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
set PATH=%CUDA_PATH%\bin\x64;%PATH%

echo CUDA PATH = %CUDA_PATH%
where nvrtc64_130_0.dll

REM --- cuTile Environment Variables ---
REM Enable crash dump for TileCompilerExecutionError or TileCompilerTimeoutError
set CUDA_TILE_ENABLE_CRASH_DUMP=1

REM Set compiler timeout (in seconds) - adjust as needed
set CUDA_TILE_COMPILER_TIMEOUT_SEC=300

REM Print cuTile Python IR during compilation (for debugging TileTypeError)
set CUDA_TILE_LOGS=CUTILEIR

REM Configure temporary files directory
set CUDA_TILE_TEMP_DIR=%TEMP%\cuda_tile

echo.
echo CUDA environment configured successfully!
echo.
echo Environment variables set:
echo   CUDA_PATH: %CUDA_PATH%
echo   CUDA_TILE_ENABLE_CRASH_DUMP: %CUDA_TILE_ENABLE_CRASH_DUMP%
echo   CUDA_TILE_COMPILER_TIMEOUT_SEC: %CUDA_TILE_COMPILER_TIMEOUT_SEC%
echo   CUDA_TILE_LOGS: %CUDA_TILE_LOGS%
echo   CUDA_TILE_TEMP_DIR: %CUDA_TILE_TEMP_DIR%
echo.
