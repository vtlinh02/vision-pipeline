#!/usr/bin/env python3
"""
Purpose: Validate ONNX with:
- onnx.checker.check_model
- ONNX Runtime smoke inference on zeros input

References:
- ONNX checker docs
- ORT Python inference usage
"""

import argparse
import numpy as np
import onnx
import onnxruntime as ort

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    print("[validate_onnx] Loading model...")
    m = onnx.load(args.model)

    print("[validate_onnx] Running onnx.checker.check_model...")
    onnx.checker.check_model(m)
    print("[validate_onnx] ONNX checker OK.")

    print("[validate_onnx] ORT smoke inference...")
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    x = np.zeros((1, 3, args.imgsz, args.imgsz), dtype=np.float32)
    outs = sess.run(None, {in_name: x})
    print("[validate_onnx] ORT OK. Output0 shape:", outs[0].shape)

if __name__ == "__main__":
    main()