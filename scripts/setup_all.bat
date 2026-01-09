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
echo  Setup Complete!
echo ========================================
echo.
echo You can now run:
echo   uv sync          - Install project dependencies
echo   uv run python    - Run Python with the project environment
echo.
