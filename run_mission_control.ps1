# =========================================================================
# AIcoScientist Discovery Mission Control - PowerShell Launcher
# =========================================================================

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  AIcoScientist Discovery Mission Control" -ForegroundColor Green
Write-Host "  Advisor Presentation System" -ForegroundColor White
Write-Host "=======================================================" -ForegroundColor Cyan

$PythonExe = "python"
if (Test-Path ".venv\Scripts\python.exe") {
    $PythonExe = ".venv\Scripts\python.exe"
}

# Ensure snapshot exists
if (-not (Test-Path "presentation\data\snapshot.json")) {
    Write-Host "Compiling deterministic presentation snapshot..." -ForegroundColor Yellow
    & $PythonExe presentation\scripts\build_snapshot.py
}

Write-Host "Launching Mission Control server at http://127.0.0.1:8501" -ForegroundColor Green
Write-Host "Press Ctrl+C to terminate server." -ForegroundColor Gray
& $PythonExe presentation\backend\server.py --port 8501
