@echo off
REM ============================================================================
REM Complete Environment Setup Script
REM ============================================================================
REM This script sets up both UV and CUDA environments
REM Usage: scripts\setup_all.bat
REM ============================================================================

echo ========================================
echo  Environment Setup
echo ========================================
echo.

REM Setup UV
echo [1/2] Setting up UV package manager...
call "%~dp0setup_uv.bat"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: UV setup failed
    exit /b 1
)

echo.
echo ========================================
echo.

REM Setup CUDA
echo [2/2] Setting up CUDA environment...
call "%~dp0setup_cuda_env.bat"

echo.
echo ========================================
echo.

REM Install dependencies
echo [3/3] Installing project dependencies...
call uv sync
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: uv sync failed, continuing...
)

REM Patch venv activation
echo.
echo Patching virtual environment with CUDA settings...
call "%~dp0patch_venv_activation.bat"

echo.
echo ========================================
echo  Setup Complete!
echo ========================================
echo.
echo CUDA environment is configured!
echo To use the project:
echo   1. Activate the virtual environment:
echo      .venv\Scripts\activate.bat
echo.
echo   2. Run your code:
echo      python src\main.py
echo.
echo   Or use uv run:
echo      uv run python src\main.py
echo.
