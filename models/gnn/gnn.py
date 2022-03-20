import torch
import torch.nn as nn
from utils.general import non_max_suppression
from torchvision.ops import RoIAlign

from .model import Model

CONF_THRES = 0.596
CELL_SIZE_LIMIT = 200

conv_output_size = 64


class GNN(nn.Module):

    def __init__(self, cell_cls):
        super().__init__()
        self.roi_align_1 = RoIAlign(
            (1), spatial_scale=1 / 8, sampling_ratio=-1)
        self.roi_align_2 = RoIAlign(
            (1), spatial_scale=1 / 16, sampling_ratio=-1)
        self.roi_align_3 = RoIAlign(
            (1), spatial_scale=1 / 32, sampling_ratio=-1)
        self.roi_align_4 = RoIAlign(
            (1), spatial_scale=1 / 64, sampling_ratio=-1)

        self.conv = nn.Sequential(
            nn.Linear(1920, conv_output_size),
            nn.LeakyReLU(),
            nn.Linear(conv_output_size, conv_output_size),
            nn.LeakyReLU(),
        )

        self.m = Model()

    def forward(self, x):
        # print(x[1].shape)  # batch, ch, h, w
        # print(x[2].shape)
        # print(x[3].shape)
        # print(x[4].shape)
        # print(x[0][0].shape) # x, y, x2, y2, score, class

        out, train_out = x[0]
        device = x[1].device
        whwh = (torch.tensor(x[1].shape[-2:]).to(device) * 8)[[1, 0, 1, 0]]
        result, cells = None, None

        if out is not None:
            preds = non_max_suppression(
                out.detach(), conf_thres=CONF_THRES)

            feats = []
            for p in preds:
                if p.shape[0] > 2:
                    p = p[p[..., 4].sort(
                        dim=-1,
                        descending=True
                    ).indices][:CELL_SIZE_LIMIT, ...]
                    bbox = p[:, :4]
                    f1 = self.roi_align_1(x[1].float(), [bbox])
                    f2 = self.roi_align_2(x[2].float(), [bbox])
                    f3 = self.roi_align_3(x[3].float(), [bbox])
                    f4 = self.roi_align_4(x[4].float(), [bbox])
                    f = torch.cat((f1, f2, f3, f4), dim=1)
                    f = f.reshape(
                            f.shape[0], f.shape[1] * f.shape[2] * f.shape[3])
                    f = self.conv(f)
                    bbox = bbox / whwh
                    f = torch.cat((bbox, f), dim=-1)
                    feats.append(f)
                else:
                    feats.append(None)

            result, cells = self.m(feats)

        train_out = (train_out, result, cells)
        out = (out, result, cells)

        return train_out if self.training else (out, train_out)
