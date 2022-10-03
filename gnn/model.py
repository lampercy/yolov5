from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchvision.ops import roi_align
from torch import Tensor


from ._gnn import _GNN
from .config import IMAGE_FEATURE_OUTPUT_SIZE
from .overrides import non_max_suppression, scale_coords

# ROI_ALIGN_SHAPE = (1)
# CONV_INPUT_SIZE = np.prod(ROI_ALIGN_SHAPE) * 1344
ROI_ALIGN_SHAPE = 1
CONV_INPUT_SIZE = ROI_ALIGN_SHAPE * 1344


def roi_align_1(
    input: Tensor,
    boxes: Union[Tensor, List[Tensor]],
) -> Tensor:
    roi_align_shape = 1
    return roi_align(
        input, boxes,
        output_size=roi_align_shape, spatial_scale=1 / 8, sampling_ratio=-1
    )


def roi_align_2(
    input: Tensor,
    boxes: Union[Tensor, List[Tensor]],
) -> Tensor:
    roi_align_shape = 1
    return roi_align(
        input, boxes,
        output_size=roi_align_shape, spatial_scale=1 / 16, sampling_ratio=-1
    )


def roi_align_3(
    input: Tensor,
    boxes: Union[Tensor, List[Tensor]],
) -> Tensor:
    roi_align_shape = 1
    return roi_align(
        input, boxes,
        output_size=roi_align_shape, spatial_scale=1 / 32, sampling_ratio=-1
    )


class GNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Linear(CONV_INPUT_SIZE, IMAGE_FEATURE_OUTPUT_SIZE),
            nn.LeakyReLU(),
            nn.Linear(IMAGE_FEATURE_OUTPUT_SIZE, IMAGE_FEATURE_OUTPUT_SIZE),
            nn.LeakyReLU(),
        )

        self.m = _GNN()

    def forward(
            self,
            out: Tensor,
            train_out: List[Tensor],
            x1: Tensor,
            x2: Tensor,
            x3: Tensor,
            shapes: List[Tuple[Tensor, List[Tensor]]],
    ) -> Tuple[
            List[Tensor],
            Tensor,
            List[Tensor],
            List[Tensor]]:

        # print(x[1].shape)  # batch, ch, h, w
        # print(x[2].shape)
        # print(x[3].shape)
        # print(x[4].shape)
        # print(x[0][0].shape) # x, y, x2, y2, score, class

        conf_thres = 0.1
        iou_thres = 0.6

        device = x1.device

        hw = (torch.tensor(x1.shape[-2:]).to(device) * 8)[[0, 1]]
        result, cells = None, None

        if out is not None:
            preds = non_max_suppression(
                out.detach(),
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                agnostic=False,
            )

            feats: List[Tensor] = []
            for i, p in enumerate(preds):
                if p.shape[0] > 2:
                    bbox = p[:, :4].clone()

                    f1 = roi_align_1(x1.float(), [bbox])
                    f2 = roi_align_2(x2.float(), [bbox])
                    f3 = roi_align_3(x3.float(), [bbox])
                    f = torch.cat((f1, f2, f3), dim=1)
                    f = f.reshape(
                            f.shape[0],
                            f.shape[1] *
                            f.shape[2] *
                            f.shape[3]
                        )
                    f = self.conv(f)

                    if shapes is not None:
                        scale_coords(
                            hw, bbox, shapes[i][0], shapes[i][1])

                        img1_shape = shapes[i][0]
                        if isinstance(img1_shape, list):
                            img1_shape = torch.tensor(img1_shape)

                        bbox = bbox / img1_shape.to(device)[[1, 0, 1, 0]]

                    f = torch.cat((bbox, f), dim=-1)

                    feats.append(f)
                else:
                    pass
                    # feats.append(None)

            result, cells = self.m(feats)

        # train_out = (train_out, result, cells)
        # out = (out, result, cells)
        # return train_out if self.training else (out, train_out)
        return (train_out, out, result, cells)
