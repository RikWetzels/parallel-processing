# ============================================================================
# Patch Virtual Environment Activation Script with CUDA Settings (PowerShell)
# ============================================================================
# This script adds CUDA environment variables to .venv activation
# Run once after creating/recreating the virtual environment
# ============================================================================

Write-Host "Patching .venv activation script with CUDA settings..." -ForegroundColor Cyan

$activateScript = ".venv\Scripts\Activate.ps1"

if (-not (Test-Path $activateScript)) {
    Write-Host "ERROR: Virtual environment not found at .venv" -ForegroundColor Red
    Write-Host "Please create the virtual environment first with: uv sync" -ForegroundColor Yellow
    exit 1
}

# Check if already patched
$content = Get-Content $activateScript -Raw
if ($content -match "CUDA 13.1 NVRTC fix") {
    Write-Host "Virtual environment already patched with CUDA settings." -ForegroundColor Green
    exit 0
}

# Backup original
Copy-Item $activateScript "$activateScript.backup"

# Add CUDA settings to activation script
$cudaSettings = @"

# --- CUDA 13.1 NVRTC fix ---
`$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
`$env:PATH = "`$env:CUDA_PATH\bin\x64;`$env:PATH"

# --- cuTile Environment Variables ---
`$env:CUDA_TILE_ENABLE_CRASH_DUMP = "1"
`$env:CUDA_TILE_COMPILER_TIMEOUT_SEC = "300"
`$env:CUDA_TILE_LOGS = "CUTILEIR"
`$env:CUDA_TILE_TEMP_DIR = "`$env:TEMP\cuda_tile"
"@

Add-Content -Path $activateScript -Value $cudaSettings

Write-Host ""
Write-Host "SUCCESS! Virtual environment activation script patched." -ForegroundColor Green
Write-Host ""
Write-Host "The following environment variables will now be set automatically" -ForegroundColor Cyan
Write-Host "when you activate the virtual environment:" -ForegroundColor Cyan
Write-Host "  - CUDA_PATH"
Write-Host "  - CUDA_TILE_ENABLE_CRASH_DUMP"
Write-Host "  - CUDA_TILE_COMPILER_TIMEOUT_SEC"
Write-Host "  - CUDA_TILE_LOGS"
Write-Host "  - CUDA_TILE_TEMP_DIR"
Write-Host ""
Write-Host "To activate: .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host ""
