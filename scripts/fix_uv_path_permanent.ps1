# ============================================================================
# Permanent UV PATH Fix Script
# ============================================================================
# This script ensures UV is ALWAYS available by:
# 1. Adding UV to user PATH environment variable
# 2. Creating a PowerShell profile that loads UV
# 3. Refreshing the current session
# ============================================================================

Write-Host "`n=== UV Permanent PATH Fix ===" -ForegroundColor Cyan
Write-Host "This will ensure UV is always available in all terminals.`n" -ForegroundColor White

# Define UV path
$uvPath = "$env:USERPROFILE\.local\bin"
$uvExe = "$uvPath\uv.exe"

# Step 1: Verify UV is installed
Write-Host "[1/4] Checking UV installation..." -ForegroundColor Yellow
if (-not (Test-Path $uvExe)) {
    Write-Host "ERROR: UV not found at $uvPath" -ForegroundColor Red
    Write-Host "Please install UV first using:" -ForegroundColor Yellow
    Write-Host "  powershell -c `"irm https://astral.sh/uv/install.ps1 | iex`"" -ForegroundColor Cyan
    exit 1
}
Write-Host "  [OK] UV found at $uvPath" -ForegroundColor Green

# Step 2: Add to user PATH (permanent)
Write-Host "`n[2/4] Adding UV to user PATH environment variable..." -ForegroundColor Yellow
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$uvPath*") {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$uvPath", "User")
    Write-Host "  [OK] UV added to user PATH permanently!" -ForegroundColor Green
} else {
    Write-Host "  [OK] UV already in user PATH" -ForegroundColor Green
}

# Step 3: Create/Update PowerShell profile
Write-Host "`n[3/4] Setting up PowerShell profile..." -ForegroundColor Yellow
$profilePath = $PROFILE
$profileDir = Split-Path -Parent $profilePath

# Create profile directory if it doesn't exist
if (-not (Test-Path $profileDir)) {
    New-Item -ItemType Directory -Path $profileDir -Force | Out-Null
    Write-Host "  [OK] Created profile directory: $profileDir" -ForegroundColor Green
}

# Profile content to add
$profileContent = @"

# Auto-load UV into PATH
`$uvPath = "`$env:USERPROFILE\.local\bin"
if ((Test-Path `$uvPath) -and (`$env:Path -notlike "*`$uvPath*")) {
    `$env:Path = "`$uvPath;`$env:Path"
}
"@

# Check if profile exists and if it already has UV setup
$addToProfile = $true
if (Test-Path $profilePath) {
    $existingContent = Get-Content -Path $profilePath -Raw
    if ($existingContent -like "*\.local\bin*" -or $existingContent -like "*uv*") {
        Write-Host "  [OK] PowerShell profile already configured" -ForegroundColor Green
        $addToProfile = $false
    }
}

if ($addToProfile) {
    Add-Content -Path $profilePath -Value $profileContent
    Write-Host "  [OK] Added UV auto-load to PowerShell profile" -ForegroundColor Green
    Write-Host "    Profile location: $profilePath" -ForegroundColor Gray
}

# Step 4: Load UV in current session
Write-Host "`n[4/4] Loading UV in current session..." -ForegroundColor Yellow
$env:Path = "$uvPath;$env:Path"
Write-Host "  [OK] UV loaded in current session" -ForegroundColor Green

# Verify it works
Write-Host "`n=== Verification ===" -ForegroundColor Cyan
try {
    $version = & uv --version
    Write-Host "  [OK] UV is working: $version" -ForegroundColor Green
} catch {
    Write-Host "  [ERROR] UV still not available" -ForegroundColor Red
    exit 1
}

Write-Host "`n=== SUCCESS ===" -ForegroundColor Green
Write-Host "UV is now permanently configured!" -ForegroundColor White
Write-Host "`nWhat was done:" -ForegroundColor Cyan
Write-Host "  1. [OK] UV added to user PATH environment variable" -ForegroundColor White
Write-Host "  2. [OK] PowerShell profile configured to auto-load UV" -ForegroundColor White
Write-Host "  3. [OK] UV loaded in current session" -ForegroundColor White
Write-Host "`nNOTE:" -ForegroundColor Yellow
Write-Host "  - New terminals will automatically have UV in PATH" -ForegroundColor White
Write-Host "  - Current VS Code terminals: Close and reopen them" -ForegroundColor White
Write-Host "  - You can now use UV commands anywhere" -ForegroundColor White
Write-Host ""
