from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .config import NUM_OF_CLASS, IMAGE_FEATURE_OUTPUT_SIZE


NEAREST_K = 8
INPUT_SIZE = IMAGE_FEATURE_OUTPUT_SIZE + 4
EDGE_OUTPUT_SIZE = 32
FEATURE_SIZE = INPUT_SIZE * 2 + EDGE_OUTPUT_SIZE * 4
LINEAR_EMBEDDING = 32


class _GNN(nn.Module):
    def __init__(self):
        super(_GNN, self).__init__()

        self.INPUT_SIZE = INPUT_SIZE
        self.k = NEAREST_K

        self.bn = nn.BatchNorm1d(INPUT_SIZE)

        self.linear = nn.Sequential(
            nn.BatchNorm1d(FEATURE_SIZE),
            nn.Linear(FEATURE_SIZE, LINEAR_EMBEDDING),
            nn.LeakyReLU(),
            nn.Linear(LINEAR_EMBEDDING, LINEAR_EMBEDDING),
            nn.LeakyReLU(),
            nn.Linear(LINEAR_EMBEDDING, NUM_OF_CLASS),
        )

        self.edge_conv1 = nn.Sequential(
            nn.Linear(self.INPUT_SIZE * self.k, EDGE_OUTPUT_SIZE),
            nn.LeakyReLU(),
            nn.Linear(EDGE_OUTPUT_SIZE, EDGE_OUTPUT_SIZE),
            nn.LeakyReLU(),
        )

        self.edge_conv2 = nn.Sequential(
            nn.Linear(EDGE_OUTPUT_SIZE * self.k, EDGE_OUTPUT_SIZE),
            nn.LeakyReLU(),
            nn.Linear(EDGE_OUTPUT_SIZE, EDGE_OUTPUT_SIZE),
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

    def _forward(self, feat0: List[Tensor]):
        feat1: List[Tensor] = []
        for f in feat0:
            if f is not None:
                feat1.append(f)

        if feat1:
            feat1 = self.batch_norm_feat(feat1)
            feat1 = [self.get_feature_combination(x) for x in feat1]

            feat2 = torch.cat(feat1, 0)  # -> (b*n, p)
            feat2 = self.linear(feat2)

            return feat2

    def forward(
        self,
        feat: List[Tensor],
    ) -> Tuple[List[Tensor], List[Tensor]]:
        x = self._forward(feat)

        cursor = 0
        cells: List[Tensor] = []
        preds: List[Tensor] = []

        for f in feat:
            if f is None or x is None:
                pass
                # cells.append(None)
                # preds.append(None)
            else:
                length = int(f.shape[0] * (f.shape[0] - 1) / 2)
                y = x[cursor:cursor + length]
                cursor += length
                cells.append(f[:, :5])
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

    def get_feature_combination(self, x: Tensor):
        device = x.device
        x = torch.cat((x, self.extract_edge_feature(x)), -1)
        n = x.shape[0]

        x = x.unsqueeze(1)  # (n, p) -> (n, 1, p)
        x = x.expand(-1, n, -1)  # (n, 1, p) -> (n, n, p)
        x = torch.cat((x, x.transpose(0, 1)), -1)

        p = x.shape[2]

        mask = torch.ones((n, n)).triu(diagonal=1)  # (n, n)
        mask = mask.unsqueeze(-1)  # (n, n) -> (n, n, 1)
        mask = mask.expand(-1, -1, p)  # (n, n, 1) -> (n, n, p)
        mask = mask.to(device)
        x = x.masked_select(mask > 0)  # (n, n, 4) -> (n * (n - 1) / 2 * p)
        x = x.view((-1, p))  # (n * (n - 1) / 2 * p) -> (n * (n - 1) / 2, p)

        return x
