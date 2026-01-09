@echo off
REM ============================================================================
REM UV Package Manager Setup Script
REM ============================================================================
REM This script checks for UV installation and adds it to PATH
REM Usage: scripts\setup_uv.bat
REM ============================================================================

echo Checking for UV installation...

REM Check if uv is already in PATH
where uv >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo UV is already available in PATH
    uv --version
    goto :end
)

REM Check if uv exists in default installation location
set UV_PATH=%USERPROFILE%\.local\bin

if exist "%UV_PATH%\uv.exe" (
    echo UV found at %UV_PATH%
    echo Adding UV to PATH for this session...
    set "PATH=%UV_PATH%;%PATH%"
    uv --version
    echo.
    echo UV is now available for this terminal session.
    goto :end
)

REM UV not found - provide installation instructions
echo.
echo UV not found on this system.
echo.
echo To install UV, run the following command:
echo.
echo   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
echo.
echo After installation, run this script again or restart your terminal.
echo.
exit /b 1

:end
echo.
echo UV setup complete!
echo You can now use: uv sync, uv add, uv run, etc.
echo.
