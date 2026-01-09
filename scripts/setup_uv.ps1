# ============================================================================
# UV Package Manager Setup Script (PowerShell)
# ============================================================================
# This script installs UV if needed and adds it to PATH
# Usage: .\scripts\setup_uv.ps1
# ============================================================================

Write-Host "Checking for UV installation..." -ForegroundColor Cyan

# Check if uv is already in PATH
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($uvCommand) {
    Write-Host "UV is already available in PATH" -ForegroundColor Green
    uv --version
    exit 0
}

# Check default installation location
$uvPath = "$env:USERPROFILE\.local\bin"
$uvExe = "$uvPath\uv.exe"

if (Test-Path $uvExe) {
    Write-Host "UV found at $uvPath" -ForegroundColor Green
    Write-Host "Adding UV to PATH for this session..." -ForegroundColor Yellow
    $env:Path = "$uvPath;$env:Path"
    uv --version
    Write-Host "`nUV is now available for this PowerShell session." -ForegroundColor Green
    
    # Ask if user wants to add to PATH permanently
    Write-Host "`nWould you like to add UV to your user PATH permanently? (Y/N)" -ForegroundColor Yellow
    $response = Read-Host
    if ($response -eq 'Y' -or $response -eq 'y') {
        $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
        if ($userPath -notlike "*$uvPath*") {
            [Environment]::SetEnvironmentVariable(
                "Path",
                "$userPath;$uvPath",
                "User"
            )
            Write-Host "UV added to user PATH permanently!" -ForegroundColor Green
            Write-Host "Please restart your terminal for the change to take effect." -ForegroundColor Yellow
        } else {
            Write-Host "UV is already in user PATH." -ForegroundColor Green
        }
    }
    exit 0
}

# UV not found - install it
Write-Host "`nUV not found. Installing UV..." -ForegroundColor Yellow
Write-Host "Running official UV installer..." -ForegroundColor Cyan

try {
    Invoke-Expression (Invoke-RestMethod https://astral.sh/uv/install.ps1)
    
    # Add to PATH for current session
    if (Test-Path $uvExe) {
        $env:Path = "$uvPath;$env:Path"
        Write-Host "`nUV installed successfully!" -ForegroundColor Green
        uv --version
        
        # Ask about permanent PATH
        Write-Host "`nWould you like to add UV to your user PATH permanently? (Y/N)" -ForegroundColor Yellow
        $response = Read-Host
        if ($response -eq 'Y' -or $response -eq 'y') {
            $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
            if ($userPath -notlike "*$uvPath*") {
                [Environment]::SetEnvironmentVariable(
                    "Path",
                    "$userPath;$uvPath",
                    "User"
                )
                Write-Host "UV added to user PATH permanently!" -ForegroundColor Green
                Write-Host "Please restart your terminal for the change to take effect." -ForegroundColor Yellow
            }
        }
    }
} catch {
    Write-Host "`nError installing UV: $_" -ForegroundColor Red
    Write-Host "Please visit https://docs.astral.sh/uv/ for manual installation." -ForegroundColor Yellow
    exit 1
}

Write-Host "`nUV setup complete!" -ForegroundColor Green
Write-Host "You can now use: uv sync, uv add, uv run, etc." -ForegroundColor Cyan
