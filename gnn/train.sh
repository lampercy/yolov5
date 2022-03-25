#!/bin/sh
python train.py --img 640 --data $1 --batch 16 --epochs $3 --weights $2 --workers 1 --cfg ./gnn/yolov5m6_gnn.yaml --hyp ./gnn/hyp.yaml --noautoanchor --freeze 34
