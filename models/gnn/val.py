import statistics
from collections import defaultdict
from models.gnn.loss import (
    match_predicted_cells_with_targets,
    filter_tri_matrix_by_indices,
)

import torch

NUM_OF_CLASS = 2


def get_confusion_matrix(
        preds, predicted_cells, gnn_targets, confusion_matrix):

    if not [p for p in preds if p is not None]:
        return confusion_matrix

    device = [p for p in preds if p is not None][0].device

    for pred, cells, gnn_target in zip(preds, predicted_cells, gnn_targets):
        cell_label, cls_label = gnn_target
        cell_label = cell_label.to(device)
        cls_label_count = cls_label.shape[0]

        if pred is not None:
            cls_label = torch.stack(
                (cls_label == 1, cls_label == 2), dim=-1).float().to(device)

            cell_indices, label_indices = match_predicted_cells_with_targets(
                cells, cell_label, device
            )

            if cell_indices.shape[0] > 1 and label_indices.shape[0] > 1:
                x = filter_tri_matrix_by_indices(
                    cell_indices, pred, device)

                y = filter_tri_matrix_by_indices(
                    label_indices, cls_label, device)

                for cls in range(NUM_OF_CLASS):
                    TP, TN, FP, FN = get_confusion_matrix_by_cls(
                        x.sigmoid(), y, cls)
                    FN += cls_label_count - x.shape[0]

                    for score, name in zip(
                            [TP, TN, FP, FN], ['TP', 'TN', 'FP', 'FN']):
                        confusion_matrix[cls][name] += score

        else:
            for cls in range(NUM_OF_CLASS):
                confusion_matrix[cls]['FN'] += cls_label_count

    return confusion_matrix


def get_confusion_matrix_by_cls(pred, truth, cls):
    device = pred.device

    pred = pred[:, cls].float()
    truth = truth[:, cls].float()

    TP = torch.sum(pred * truth)
    TN = torch.sum(
        (torch.ones(pred.shape).to(device) - pred) *
        (torch.ones(truth.shape).to(device) - truth))
    FP = torch.sum(
        pred * (torch.ones(pred.shape).to(device) - truth))
    FN = torch.sum(
        (torch.ones(pred.shape).to(device) - pred) * truth)

    return [x.item() for x in [TP, TN, FP, FN]]


def compute_confusion_matrix_scores(confusion_matrix):
    result = {}
    scores = defaultdict(list)

    for cls in range(NUM_OF_CLASS):
        total_TP = confusion_matrix[cls]['TP']
        total_TN = confusion_matrix[cls]['TN']
        total_FP = confusion_matrix[cls]['FP']
        total_FN = confusion_matrix[cls]['FN']

        precision = (total_TP / (total_TP + total_FP)) \
            if total_TP + total_FP != 0 else 0
        recall = (total_TP / (total_TP + total_FN)) \
            if total_TP + total_FN != 0 else 0
        f1 = statistics.harmonic_mean((recall, precision))

        result = {
            **result,
            f'TP_{cls}': total_TP,
            f'FP_{cls}': total_FP,
            f'FN_{cls}': total_FN,
            f'TN_{cls}': total_TN,
            f'precision_{cls}': precision,
            f'recall_{cls}': recall,
            f'f1_{cls}': f1,
        }

        scores['f1'].append(f1)
        scores['precision'].append(precision)
        scores['recall'].append(recall)

    result = {
        **result,
        'precision': statistics.harmonic_mean(scores['precision']),
        'recall': statistics.harmonic_mean(scores['recall']),
        'f1': statistics.harmonic_mean(scores['f1']),
    }

    return result
