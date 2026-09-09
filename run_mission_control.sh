#!/usr/bin/env bash
# =========================================================================
# AIcoScientist Discovery Mission Control - Linux/macOS Launcher
# =========================================================================

set -e

echo "Starting AIcoScientist Discovery Mission Control..."

PYTHON_EXE="python3"
if [ -f ".venv/bin/python" ]; then
    PYTHON_EXE=".venv/bin/python"
elif [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON_EXE=".venv/Scripts/python.exe"
fi

if [ ! -f "presentation/data/snapshot.json" ]; then
    echo "Compiling deterministic presentation snapshot..."
    $PYTHON_EXE presentation/scripts/build_snapshot.py
fi

echo "Launching Mission Control at http://127.0.0.1:8501"
$PYTHON_EXE presentation/backend/server.py --port 8501
