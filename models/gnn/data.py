import json
from pathlib import Path
import torch


def get_gnn_label(image_path):
    p = Path(image_path)
    parts = list(p.parts)
    parts[-3] = 'dgcnn'
    with Path(*parts).with_suffix('.json').open() as f:
        data = json.load(f)
        cells = torch.tensor(data['cells'])
        classes = torch.tensor(data['classes'])

    return cells, classes
