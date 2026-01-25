#!/bin/bash
# Run open category dataset processing script

# Do not use set -e, as some scripts may return non-zero exit codes that don't indicate failure

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Detect Python in current environment
# Prefer python (conda environments usually prioritize this)
if command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
else
    echo "Error: Python not found"
    exit 1
fi

echo "Using Python: $PYTHON_CMD ($($PYTHON_CMD --version 2>&1))"
echo "Python path: $(command -v $PYTHON_CMD)"
echo ""

echo "=========================================="
echo "Starting open category dataset processing"
echo "=========================================="
echo ""

# Run the processing script
SCRIPT_PATH="datasets_process/process_open.py"

if [ -f "$SCRIPT_PATH" ]; then
    echo "----------------------------------------"
    echo "Running: $SCRIPT_PATH"
    echo "----------------------------------------"
    "$PYTHON_CMD" "$SCRIPT_PATH"
    echo ""
else
    echo "Error: File does not exist: $SCRIPT_PATH"
    exit 1
fi

echo ""
echo "=========================================="
echo "Open category dataset processing completed"
echo "=========================================="
