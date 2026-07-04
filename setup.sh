#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# setup.sh -- Cricket Bowling AI Coach setup
# Installs Python dependencies.
# The MediaPipe pose model is now bundled inside the mediapipe package --
# no separate download is required.
# -----------------------------------------------------------------------------

set -e

echo ""
echo "Cricket Bowling AI Coach -- Setup"
echo "----------------------------------"

# Install Python dependencies
echo ""
echo "[INFO] Installing Python dependencies..."
pip install -r requirements.txt --quiet

echo ""
echo "[OK]  Setup complete!"
echo ""
echo "Run the app with:"
echo "    streamlit run app.py"
echo ""
