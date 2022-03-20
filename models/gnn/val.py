import statistics
from collections import defaultdict

import torch

NUM_OF_CLASS = 2


def get_confusion_matrix(preds, predicted_cells, gnn_targets):
    for pred, cells, gnn_target in zip(preds, predicted_cells, gnn_targets):
        print(pred.shape)
        print(cells.shape)
        print(gnn_target.shape)


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


def compute_scores(confusion_matrix):
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
