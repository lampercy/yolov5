import os
import json

import numpy as np
import torch

from typing import NamedTuple, Tuple, List
from PIL.Image import Image
from PIL import Image as PILImage

from utils.general import LOGGER, check_requirements, colorstr, file_size


LETTERBOX_FILL_COLOR = (114, 114, 114)


class ImageTensorData(NamedTuple):
    image: np.ndarray
    dx: float
    dy: float
    width: int
    height: int
    img0_size: List[float]


def resize_image(
    im: Image,
    upscaling: bool,
    target_size: int,
    resize_to_shorter_side: bool = False,
) -> Tuple[Image, float]:

    width = im.width
    height = im.height

    if resize_to_shorter_side:
        shorter_side = min(width, height)
        resize_factor = target_size / shorter_side
    else:
        longer_side = max(width, height)
        resize_factor = target_size / longer_side

    if resize_factor >= 1 and not upscaling:
        return im, 1

    resized = im.resize(
        (int(width * resize_factor), int(height * resize_factor)),
        PILImage.LANCZOS,
    )

    return resized, resize_factor


def letterbox(
    im: Image, target_size: int
) -> Tuple[Image, Tuple[float, float]]:

    assert im.width <= target_size
    assert im.height <= target_size

    dx = (target_size - im.width) / 2
    dy = (target_size - im.height) / 2
    result = PILImage.new(
        im.mode, (target_size, target_size), LETTERBOX_FILL_COLOR
    )
    result.paste(im, (int(dx), int(dy)))

    return result, (dx, dy)


def convert_pil_image_to_tensor(image: Image, image_tensor_dtype: np.dtype):
    image = np.array(image)[:, :, ::-1]
    image = torch.as_tensor(
        image.astype(image_tensor_dtype).transpose(2, 0, 1)
    )
    return image


def load_image_as_tensor(
    image: Image, target_size: int, image_tensor_dtype: np.dtype = np.float32
) -> ImageTensorData:

    img0_size = image.size
    image, _ = resize_image(image, False, target_size)
    width, height = image.size
    image, (dx, dy) = letterbox(image, target_size)
    image = convert_pil_image_to_tensor(image, image_tensor_dtype)
    image /= 255.0

    return ImageTensorData(
        image, dx, dy, width, height, img0_size)


def prepare_img_data_for_export(
        target_size, img_src=os.path.join(
            os.path.dirname(__file__), 'sample-image-for-export.jpg')):
    img = PILImage.open(img_src)

    data = load_image_as_tensor(img, target_size)

    img = data.image.unsqueeze(0)

    h, w = data.image.shape[1:]
    w0, h0 = data.img0_size
    pad = torch.tensor([data.dy, data.dx])
    shapes = [[torch.tensor([h0, w0]), [torch.tensor([h / h0, w / w0]), pad]]]

    return img, shapes


def export_gnn_onnx(
        model, file, opset, train, dynamic,
        simplify, target_size, prefix=colorstr('ONNX:')):
    # YOLOv5 ONNX export
    check_requirements(('onnx',))
    import onnx

    im, shapes = prepare_img_data_for_export(target_size)

    LOGGER.info(f'\n{prefix} starting export with onnx {onnx.__version__}...')
    f = file.with_suffix('.onnx')

    torch.onnx.export(model, (im, shapes), f, verbose=False, opset_version=opset,
                      training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
                      do_constant_folding=not train,
                      input_names=['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6'],
                      output_names=['output'],
                      dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                                    'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
                                    } if dynamic else None)

    # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    # LOGGER.info(onnx.helper.printable_graph(model_onnx.graph))  # print

    # Simplify
    if simplify:
        try:
            check_requirements(('onnx-simplifier',))
            import onnxsim

            LOGGER.info(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=dynamic,
                input_shapes={'images': list(im.shape)} if dynamic else None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, f)
        except Exception as e:
            LOGGER.info(f'{prefix} simplifier failure: {e}')
    LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    return f


def export_torchscript_gnn(model, file, target_size, prefix=colorstr('TorchScript:')):
    LOGGER.info(f'\n{prefix} starting export with torch {torch.__version__}...')
    f = file.with_suffix('.torchscript')

    im, shapes = prepare_img_data_for_export(target_size)

    ts = torch.jit.trace(model, (im, [tuple(shape) for shape in shapes]), strict=False)

    d = {"shape": im.shape, "stride": int(max(model.stride)), "names": model.names}
    extra_files = {'config.txt': json.dumps(d)}  # torch._C.ExtraFilesMap()
    ts.save(str(f), _extra_files=extra_files)

    LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    return f
