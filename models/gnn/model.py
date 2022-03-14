import math
from itertools import combinations

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


NUM_OF_CLASS = 2
NEAREST_K = 8

input_size = 64 + 4
edge_output_size = 32
feature_size = input_size * 2 + edge_output_size * 4
linear_embedding = 32

IOU_THRES = 0.1


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.input_size = input_size
        self.k = NEAREST_K

        self.bn = nn.BatchNorm1d(input_size)

        self.linear = nn.Sequential(
            nn.BatchNorm1d(feature_size),
            nn.Linear(feature_size, linear_embedding),
            nn.LeakyReLU(),
            nn.Linear(linear_embedding, linear_embedding),
            nn.LeakyReLU(),
            nn.Linear(linear_embedding, NUM_OF_CLASS),
        )

        self.edge_conv1 = nn.Sequential(
            nn.Linear(self.input_size * self.k, edge_output_size),
            nn.LeakyReLU(),
            nn.Linear(edge_output_size, edge_output_size),
            nn.LeakyReLU(),
        )

        self.edge_conv2 = nn.Sequential(
            nn.Linear(edge_output_size * self.k, edge_output_size),
            nn.LeakyReLU(),
            nn.Linear(edge_output_size, edge_output_size),
            nn.LeakyReLU(),
        )

    def batch_norm_feat(self, feat: List[Tensor]):
        x = torch.cat(feat, 0)
        x = self.bn(x)  # (n*b, p)

        counter = 0
        results = []
        for f in feat:
            results.append(x[counter:counter + f.shape[0], :])
            counter += f.shape[0]

        return results

    def _forward(self, feat):
        feat = [f for f in feat if f is not None]
        if feat:
            feat = self.batch_norm_feat(feat)
            feat = [self.get_feature_combination(x) for x in feat]

            x = torch.cat(feat, 0)  # -> (b*n, p)
            x = self.linear(x)

            return x

    def forward(
        self,
        feat: List[Tensor],
    ) -> List[Tuple[Optional[Tensor], Optional[Tensor]]]:
        x = self._forward(feat)

        cursor = 0
        cells = []
        preds = []

        for f in feat:
            if f is None or x is None:
                cells.append(None)
                preds.append(None)
            else:
                length = int(f.shape[0] * (f.shape[0] - 1) / 2)
                y = x[cursor:cursor + length]
                cursor += length
                cells.append(f[:, :4])
                preds.append(y)

        return preds, cells

    def get_closest_cells(self, feat: Tensor):
        n = feat.shape[0]
        x = feat[:, :4].unsqueeze(1)  # (n, p) -> (n, 1, p)
        x = x.expand(-1, n, -1)  # (n, 1, p) -> (n, n, p)
        t = x.transpose(0, 1)  # (n, n, p) -> (n, n, p)
        x = (x - t).abs().sum(-1)  # (n, n)
        k = min(x.shape[0], self.k)
        i = x.topk(k, largest=False).indices  # (n, k)

        return i

    def extract_edge_feature(self, feat: Tensor):
        device = feat.device

        i = self.get_closest_cells(feat)

        edge1 = feat[i]  # (n, p) -> (n, k, p)
        edge1 = edge1.transpose(1, 2)  # (n, k, p) -> (n, p, k)
        zeros = torch.zeros(edge1.shape[0], edge1.shape[1], self.k)
        zeros[:, :, :edge1.shape[2]] = edge1
        edge1 = zeros.to(device)  # (n, p, k)

        edge1 = edge1.view(edge1.shape[0], -1)  # (n, p * k)
        edge1 = self.edge_conv1(edge1)  # (n, l)

        edge2 = edge1[i]  # (n, k, l)
        edge2 = edge2.transpose(1, 2)  # (n, k, l) -> (n, l, k)
        zeros = torch.zeros(edge2.shape[0], edge2.shape[1], self.k)
        zeros[:, :, :edge2.shape[2]] = edge2
        edge2 = zeros.to(device)  # (n, l, k)

        edge2 = edge2.view(edge2.shape[0], -1)  # (n, l * k)
        edge2 = self.edge_conv2(edge2)  # (n, l)

        return torch.cat((edge1, edge2), -1)

    def get_feature_combination(self, feat: Tensor):
        x = feat
        device = feat.device
        edge = self.extract_edge_feature(feat)
        x = torch.cat((x, edge), -1)
        n = x.shape[0]

        x = x.unsqueeze(1)  # (n, p) -> (n, 1, p)
        x = x.expand(-1, n, -1)  # (n, 1, p) -> (n, n, p)
        t = x.transpose(0, 1)  # (n, n, p) -> (n, n, p)
        x = torch.cat((x, t), -1)

        p = x.shape[2]

        mask = torch.ones((n, n)).triu(diagonal=1)  # (n, n)
        mask = mask.unsqueeze(-1)  # (n, n) -> (n, n, 1)
        mask = mask.expand(-1, -1, p)  # (n, n, 1) -> (n, n, p)
        mask = mask.to(device)
        x = x.masked_select(mask > 0)  # (n, n, 4) -> (n * (n - 1) / 2 * p)
        x = x.view((-1, p))  # (n * (n - 1) / 2 * p) -> (n * (n - 1) / 2, p)

        return x


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


def cal_loss_by_cls(
        cell_indices, label_indices, cls_label, pred, device, cls_label_count):
    loss = nn.BCEWithLogitsLoss(reduction='none')

    x = filter_tri_matrix_by_indices(
        cell_indices, pred, device)
    y = filter_tri_matrix_by_indices(
        label_indices, cls_label, device)

    loss_by_cls = loss(x, y)
    return (
        torch.tensor(1).to(device) *
        (cls_label_count - loss_by_cls.shape[-1]) +
        loss_by_cls.sum() / 2) / cls_label_count


def cal_gnn_loss(preds, predicted_cells, gnn_targets, device):
    result = torch.tensor(0, dtype=torch.float).to(device)

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

    return result

def find_area(cell):
    return F.relu(
        cell[..., 2] - cell[..., 0]) * F.relu(cell[..., 3] - cell[..., 1])


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
        dim=-1,
    )

    return result
