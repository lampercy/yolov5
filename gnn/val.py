import statistics
from collections import defaultdict

import torch

from utils.loggers import wandb

from .loss import (
    match_predicted_cells_with_truths,
    filter_tri_matrix_by_indices,
)
from .config import NUM_OF_CLASS


class GNNConfusionMatrix:

    def __init__(self, device):
        self.device = device
        self.container = defaultdict(lambda: defaultdict(float))
        self.scores = None

    def append(
        self,
        preds,
        predicted_cells,
        gnn_targets,
    ):

        for pred, cells, gnn_target in zip(
                preds, predicted_cells, gnn_targets):

            cell_label, cls_label = gnn_target
            cell_label = cell_label.to(self.device)
            cls_label_count = cls_label.shape[0]

            if pred is not None:
                cls_label = torch.stack(
                    (cls_label == 1, cls_label == 2), dim=-1
                    ).float().to(self.device)

                cell_indices, label_indices = \
                    match_predicted_cells_with_truths(
                        cells, cell_label, self.device
                    )

                if cell_indices.shape[0] > 1 and label_indices.shape[0] > 1:
                    x = filter_tri_matrix_by_indices(
                        cell_indices, pred, self.device)

                    y = filter_tri_matrix_by_indices(
                        label_indices, cls_label, self.device)

                    for cls in range(NUM_OF_CLASS):
                        TP, TN, FP, FN = get_confusion_matrix_by_cls(
                            x.sigmoid(), y, cls)
                        FN_cell = cls_label_count - x.shape[0]

                        for score, name in zip(
                                [TP, TN, FP, FN, FN_cell],
                                ['TP', 'TN', 'FP', 'FN', 'FN_cell']):
                            self.container[cls][name] += score

            else:
                for cls in range(NUM_OF_CLASS):
                    self.container[cls]['FN'] += cls_label_count

    def compute_scores(self):
        result = {}
        scores = defaultdict(list)

        for cls in range(NUM_OF_CLASS):
            total_TP = self.container[cls]['TP']
            total_TN = self.container[cls]['TN']
            total_FP = self.container[cls]['FP']
            total_FN = self.container[cls]['FN']
            total_FN_cell = self.container[cls]['FN_cell']

            precision = (total_TP / (total_TP + total_FP)) \
                if total_TP + total_FP != 0 else 0

            recall_gnn = (total_TP / (total_TP + total_FN)) \
                if total_TP + total_FN != 0 else 0
            f1_gnn = statistics.harmonic_mean((recall_gnn, precision))

            recall = (total_TP / (total_TP + total_FN + total_FN_cell)) \
                if total_TP + total_FN != 0 else 0
            f1 = statistics.harmonic_mean((recall, precision))

            result = {
                **result,
                f'TP_{cls}': total_TP,
                f'FP_{cls}': total_FP,
                f'FN_{cls}': total_FN,
                f'TN_{cls}': total_TN,
                f'FN_cell_{cls}': total_FN_cell,
                f'precision_{cls}': precision,
                f'recall_{cls}': recall,
                f'f1_{cls}': f1,
                f'recall_gnn_{cls}': recall_gnn,
                f'f1_gnn_{cls}': f1_gnn,
            }

            scores['precision'].append(precision)
            scores['recall'].append(recall)
            scores['f1'].append(f1)
            scores['f1_gnn'].append(f1_gnn)
            scores['recall_gnn'].append(recall_gnn)

        result = {
            **result,
            'precision': statistics.harmonic_mean(scores['precision']),
            'recall': statistics.harmonic_mean(scores['recall']),
            'f1': statistics.harmonic_mean(scores['f1']),
            'recall_gnn': statistics.harmonic_mean(scores['recall_gnn']),
            'f1_gnn': statistics.harmonic_mean(scores['f1_gnn']),
        }

        self.scores = result
        return result

    def log_wandb(self, epoch):
        # here
        if wandb:
            wandb.log({
                'epoch': epoch,
                **{f'gnn/{k}': v for k, v in self.scores.items()}
            })


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
