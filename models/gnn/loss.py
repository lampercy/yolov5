import torch
import math
from itertools import combinations
import torch.nn.functional as F
import torch.nn as nn


IOU_THRES = 0.6
NUM_OF_CLASS = 2


def cal_gnn_loss(preds, predicted_cells, gnn_targets, device):
    result = torch.tensor(0, dtype=torch.float, requires_grad=True).to(device)

    for pred, cells, gnn_target in zip(preds, predicted_cells, gnn_targets):
        cell_label, cls_label = gnn_target
        cls_label_count = cls_label.shape[0]
        loss = torch.tensor(1).to(device)
        if pred is not None:
            cell_label = cell_label.to(device)
            cls_label = torch.stack(
                (cls_label == 1, cls_label == 2), dim=-1).float().to(device)

            cell_indices, label_indices = match_predicted_cells_with_targets(
                cells, cell_label, device
            )

            if cell_indices.shape[0] > 1 and label_indices.shape[0] > 1:
                loss = cal_loss_by_cls(
                    cell_indices, label_indices,
                    cls_label, pred, device, cls_label_count)

        result += loss

    return result.unsqueeze(-1)


def cal_loss_by_cls(
        cell_indices, label_indices, cls_label, pred, device, cls_label_count):
    loss = nn.BCEWithLogitsLoss(reduction='none')

    x = filter_tri_matrix_by_indices(
        cell_indices, pred, device)
    y = filter_tri_matrix_by_indices(
        label_indices, cls_label, device)

    pred_count = x.shape[0]
    remaining_count = cls_label_count - pred_count

    loss_by_cls = loss(x, y).sum() / NUM_OF_CLASS

    remaining_loss = torch.tensor(1).to(device) \
        * remaining_count

    result = (loss_by_cls + remaining_loss) / cls_label_count
    return result


def filter_tri_matrix_by_indices(x, matrix, device):
    n = int((math.sqrt(matrix.shape[0] * 8 + 1) + 1) / 2)

    def convert_index(i, j):
        if i == j:
            return torch.zeros((matrix.shape[-1])).to(device)
        a = min(i, j)
        b = max(i, j)

        index = sum(range(n - a, n)) + b - a - 1
        return matrix[index]

    result = torch.stack(
        [convert_index(i, j) for i, j in combinations(x, 2)],
        dim=0,
    )

    return result


def match_predicted_cells_with_targets(cells, cell_label, device):
    cell_label_expanded = cell_label.unsqueeze(-2).repeat(
        1, cells.shape[0], 1)

    cells_expanded = cells.unsqueeze(0).repeat(
        cell_label.shape[0], 1, 1)
    combined = torch.cat((cell_label_expanded, cells_expanded), dim=-1)
    label_area = find_area(combined[..., :4])
    cell_label_area = find_area(combined[..., 4:])
    intersection = torch.stack((
        combined[..., [0, 4]].max(dim=-1).values,
        combined[..., [1, 5]].max(dim=-1).values,
        combined[..., [2, 6]].min(dim=-1).values,
        combined[..., [3, 7]].min(dim=-1).values,
    ), dim=-1)
    intersection_area = find_area(intersection)
    iou = intersection_area / (
        label_area + cell_label_area - intersection_area)

    iou_max = iou.max(dim=-1)

    label_indices = torch.stack((
        iou_max.values,
        torch.arange(iou_max.values.shape[0]).to(device)
    ), dim=-1)

    label_indices = label_indices[label_indices[:, 0] > IOU_THRES][:, 1].long()
    cell_indices = iou_max.indices[label_indices]

    return cell_indices, label_indices


def find_area(cell):
    return F.relu(
        cell[..., 2] - cell[..., 0]) * F.relu(cell[..., 3] - cell[..., 1])
