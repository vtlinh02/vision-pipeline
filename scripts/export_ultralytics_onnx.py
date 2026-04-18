#!/usr/bin/env python3
"""
Purpose: Reproducible Ultralytics export script (PT -> ONNX) with explicit settings.

This uses:
  from ultralytics import YOLO
  model.export(format="onnx", imgsz=..., opset=..., dynamic=..., simplify=..., end2end=...)

See Ultralytics export docs and config args documentation.
"""

import argparse
from pathlib import Path

def str2bool(x: str) -> bool:
    return x.lower() in ("1", "true", "yes", "y")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="e.g. yolo26n.pt (auto-download supported)")
    ap.add_argument("--out", required=True, help="output ONNX path")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dynamic", type=str, default="false")
    ap.add_argument("--simplify", type=str, default="true")
    ap.add_argument("--end2end", type=str, default="true")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)

    # Export; Ultralytics will name output based on model name.
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=str2bool(args.dynamic),
        simplify=str2bool(args.simplify),
        end2end=str2bool(args.end2end),
    )

    print(f"[DEBUG]: the onnx exported model is at: {exported}")

    # exported may be path-like; normalize.
    exported_path = Path(str(exported))
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)

    exported_path.replace(target)
    print(f"[export_ultralytics_onnx] wrote: {target}")

if __name__ == "__main__":
    main()