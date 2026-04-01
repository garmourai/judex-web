#!/bin/bash

echo "========================================"
echo "  PICKLEBALL CALIBRATION WEB TOOL"
echo "  Installation Script"
echo "========================================"
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Initialize conda
eval "$(conda shell.bash hook)"

# Check if environment exists, if not create it
if conda env list | grep -q "pickleball_web"; then
    echo "♻️  Using existing conda environment: pickleball_web"
else
    echo "📦 Creating conda environment: pickleball_web..."
    conda create -n pickleball_web python=3.10 -y
fi

echo "🔗 Activating conda environment..."
conda activate pickleball_web

echo "📥 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo ""
echo "========================================"
echo "  ✅ Installation Complete!"
echo "========================================"
echo ""
echo "To start the calibration tool, run:"
echo "  bash run.sh"
echo ""
echo "Then open http://localhost:5000 in your browser"
echo ""
