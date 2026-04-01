#!/bin/bash

echo "========================================"
echo "  PICKLEBALL CALIBRATION WEB TOOL"
echo "========================================"
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Initialize conda
eval "$(conda shell.bash hook)"

# Check if environment exists
if ! conda env list | grep -q "pickleball_web"; then
    echo "❌ Conda environment 'pickleball_web' not found!"
    echo "Please run: bash install.sh"
    exit 1
fi

echo "🔗 Activating conda environment..."
conda activate pickleball_web

echo "🚀 Starting Flask server..."
echo ""
echo "Web tool is running at: http://localhost:5000"
echo "Press Ctrl+C to stop"
echo ""

python3 app.py
