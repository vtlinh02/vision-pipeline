#!/usr/bin/env bash
# Purpose: Download ONNX Runtime prebuilt C/C++ libs for Linux x64 into third_party/onnxruntime.

set -euo pipefail

ORT_VERSION="${ORT_VERSION:-1.24.4}"
OUT_DIR="third_party"
DEST="${OUT_DIR}/onnxruntime"
ARCHIVE="onnxruntime-linux-x64-${ORT_VERSION}.tgz"

mkdir -p "${OUT_DIR}"
rm -rf "${DEST}"
mkdir -p "${DEST}"

echo "[fetch_onnxruntime] Downloading ONNX Runtime ${ORT_VERSION}..."
curl -L -o "${OUT_DIR}/${ARCHIVE}" \
  "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ARCHIVE}"

echo "[fetch_onnxruntime] Extracting..."
tar -xzf "${OUT_DIR}/${ARCHIVE}" -C "${OUT_DIR}"
mv "${OUT_DIR}/onnxruntime-linux-x64-${ORT_VERSION}"/* "${DEST}/"

echo "[fetch_onnxruntime] Done: ${DEST}"
echo "[fetch_onnxruntime] Remember to export LD_LIBRARY_PATH=${PWD}/${DEST}/lib:\$LD_LIBRARY_PATH"