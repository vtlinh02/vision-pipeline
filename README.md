<!-- Purpose: Top-level docs for building/running the Week One CPU inference demo. -->

# Week One: CPU YOLO → ONNX → C++ ONNX Runtime (Image Inference)

This repo is a minimal, production-like Week One deliverable:
- Export YOLO model to ONNX
- Run single-image inference via C++ ONNX Runtime (CPU)
- Output:
  - JSON detections
  - overlay image with boxes/labels
  - timing metrics

## Requirements (Linux x86_64)
Unspecified distribution; tested assumptions:
- GCC or Clang (latest stable)
- CMake >= 3.20
- OpenCV dev package (e.g., libopencv-dev)
- curl, tar

## Quickstart

### 1) Fetch ONNX Runtime (C/C++)
```bash
bash scripts/fetch_onnxruntime.sh
```

### 2) Export YOLO26n ONNX (end-to-end)
```bash
bash scripts/fetch_models.sh yolo26n
```

This produces:
- models/yolo26n.onnx

### 3) Build
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Recommended flags are applied automatically for Release; you can also add:
- -O3 -march=native -DNDEBUG (see CMakeLists.txt)

### 4) Run demo
```bash
export LD_LIBRARY_PATH=$PWD/third_party/onnxruntime/lib:$LD_LIBRARY_PATH

./build/bin/detect_image \
  --config configs/demo.yaml \
  --image assets/bus.jpg \
  --model models/yolo26n.onnx \
  --out_dir outputs/demo
```

Expected outputs in outputs/demo/:
- bus.detections.json
- bus.overlay.png
- bus.metrics.json

## Benchmarking
```bash
./build/bin/detect_image --model models/yolo26n.onnx --image assets/bus.jpg \
  --out_dir outputs/bench --warmup 20 --repeat 200 --threads 1

./build/bin/detect_image --model models/yolo26n.onnx --image assets/bus.jpg \
  --out_dir outputs/bench --warmup 20 --repeat 200 --threads 4
```

Metrics recorded:
- preprocess_ms
- inference_ms
- postprocess_ms
- end_to_end_ms

## Notes
- YOLO26 end-to-end ONNX output is expected to be (1,300,6): [x1,y1,x2,y2,confidence,class_id].
- For YOLO11n/YOLOv8n classic exports, this repo includes optional external NMS postprocess.

## Troubleshooting
- If binaries fail to start: ensure LD_LIBRARY_PATH includes third_party/onnxruntime/lib
- If detections look wrong: check BGR→RGB, letterbox padding, and coordinate unmapping.

## License
This repo’s code is MIT. Model weights you download may be licensed separately; review upstream licensing before redistribution.