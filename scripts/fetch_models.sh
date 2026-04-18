#!/usr/bin/env bash
# Purpose: Create venv, install export deps, fetch YOLO weights (auto-download), export to ONNX, and fetch sample image.

set -euo pipefail

MODEL="${1:-yolo26n}"     # e.g., yolo26n, yolo11n, yolov8n
IMGSZ="${IMGSZ:-640}"
OPSET="${OPSET:-17}"

mkdir -p models assets .venv

echo "[fetch_models] Creating venv..."
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install ultralytics onnx onnxruntime opencv-python

echo "[fetch_models] Exporting ${MODEL}.pt -> ONNX..."
python scripts/export_ultralytics_onnx.py \
  --model "${MODEL}.pt" \
  --out "models/${MODEL}.onnx" \
  --imgsz "${IMGSZ}" \
  --opset "${OPSET}" \
  --dynamic false \
  --simplify true \
  --end2end true

echo "[fetch_models] Fetching sample image..."
curl -L -o assets/bus.jpg "https://ultralytics.com/images/bus.jpg"

echo "[fetch_models] Validate ONNX..."
python scripts/validate_onnx.py --model "models/${MODEL}.onnx" --imgsz "${IMGSZ}"

echo "[fetch_models] Done."