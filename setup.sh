#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# setup.sh -- Cricket Bowling AI Coach setup
# Downloads the MediaPipe Pose Landmarker model and installs dependencies.
# -----------------------------------------------------------------------------

set -e

echo ""
echo "Cricket Bowling AI Coach -- Setup"
echo "----------------------------------"

# 1. Create models directory
mkdir -p models

# 2. Download MediaPipe Pose Landmarker Lite model
MODEL_URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MODEL_PATH="models/pose_landmarker_lite.task"

if [ -f "$MODEL_PATH" ]; then
    echo "[OK]  Model already present at $MODEL_PATH"
else
    echo "[INFO] Downloading MediaPipe Pose Landmarker Lite..."
    if command -v wget &> /dev/null; then
        wget -q --show-progress "$MODEL_URL" -O "$MODEL_PATH"
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar "$MODEL_URL" -o "$MODEL_PATH"
    else
        echo "[ERROR] Neither wget nor curl found. Please download manually:"
        echo "        $MODEL_URL"
        echo "        Save as: $MODEL_PATH"
        exit 1
    fi
    echo "[OK]  Model downloaded -> $MODEL_PATH"
fi

# 3. Install Python dependencies
echo ""
echo "[INFO] Installing Python dependencies..."
pip install -r requirements.txt --quiet

echo ""
echo "[OK]  Setup complete!"
echo ""
echo "Run the app with:"
echo "    streamlit run app.py"
echo ""
