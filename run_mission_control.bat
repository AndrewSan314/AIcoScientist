@echo off
REM =========================================================================
REM AIcoScientist Discovery Mission Control - Advisor Presentation Launcher
REM =========================================================================

echo Starting AIcoScientist Discovery Mission Control...

if exist ".venv\Scripts\python.exe" (
    set PYTHON_EXE=.venv\Scripts\python.exe
) else (
    set PYTHON_EXE=python
)

REM Build snapshot if missing
if not exist "presentation\data\snapshot.json" (
    echo Building deterministic presentation snapshot...
    %PYTHON_EXE% presentation\scripts\build_snapshot.py
)

REM Launch FastAPI presentation server on port 8501
echo Launching mission control at http://localhost:8501
%PYTHON_EXE% presentation\backend\server.py --port 8501
pause
