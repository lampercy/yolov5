#!/bin/sh
python export.py --weight runs/train/exp893/weights/best.pt --include onnx --opset 14 --export_gnn
