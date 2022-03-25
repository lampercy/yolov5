import torch
import math
import torch.nn.functional as F
import torch.nn as nn


IOU_THRES = 0.6
NUM_OF_CLASS = 2


def cal_gnn_loss(cls_preds, cell_preds, gnn_truths, device):
    result = torch.tensor(0, dtype=torch.float, requires_grad=True).to(device)

    for cls_pred, cell_pred, cell_truth, cls_truth in zip(
            cls_preds, cell_preds, *zip(*gnn_truths)):

        loss = torch.tensor(1).to(device)

        if cls_pred is not None:
            cls_truth_count = cls_truth.shape[0]

            cell_truth = cell_truth.to(device)
            cls_truth = torch.stack(
                (cls_truth == 1, cls_truth == 2), dim=-1).float().to(device)

            cell_pred_indices, cell_truth_indices = \
                match_predicted_cells_with_truths(
                    cell_pred, cell_truth, device
                )

            if cell_pred_indices.shape[0] > 1 and \
                    cell_truth_indices.shape[0] > 1:

                x = filter_tri_matrix_by_indices(
                    cell_pred_indices, cls_pred, device)

                y = filter_tri_matrix_by_indices(
                    cell_truth_indices, cls_truth, device)

                loss = cal_loss_by_cls(
                    x, y, cls_truth_count, device)

        result += loss

    return result.unsqueeze(-1)


def cal_loss_by_cls(x, y, cls_truth_count, device):
    loss = nn.BCEWithLogitsLoss(reduction='none')

    pred_count = x.shape[0]
    remaining_count = cls_truth_count - pred_count

    loss_by_cls = loss(x, y).sum() / NUM_OF_CLASS

    remaining_loss = torch.tensor(1).to(device) \
        * remaining_count

    result = (loss_by_cls + remaining_loss) / cls_truth_count
    return result


def filter_tri_matrix_by_indices(x, matrix, device):
    n = int((math.sqrt(matrix.shape[0] * 8 + 1) + 1) / 2)
    i = torch.combinations(x).sort(dim=-1).values
    i = ((n * 2 - i[:, 0] - 1) * i[:, 0] / 2 + i[:, 1] - i[:, 0] - 1).long()
    result = matrix[i]
    return result


def match_predicted_cells_with_truths(cell_pred, cell_truth, device):
    cell_label_expanded = cell_truth.unsqueeze(-2).repeat(
        1, cell_pred.shape[0], 1)

    cells_expanded = cell_pred.unsqueeze(0).repeat(
        cell_truth.shape[0], 1, 1)
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

    truth_indices = torch.stack((
        iou_max.values,
        torch.arange(iou_max.values.shape[0]).to(device)
    ), dim=-1)

    truth_indices = truth_indices[truth_indices[:, 0] > IOU_THRES][:, 1].long()
    pred_indices = iou_max.indices[truth_indices]

    return pred_indices, truth_indices


def find_area(cell):
    return F.relu(
        cell[..., 2] - cell[..., 0]) * F.relu(cell[..., 3] - cell[..., 1])
