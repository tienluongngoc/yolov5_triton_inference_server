import torch
import numpy as np
import typing
from yolov5_utils import non_max_suppression, scale_coords, xyxy2xywh, letterbox



def yolov5_preporocessing(img):
    img_height = 640
    img_width = 640
    device = "cuda"

    if img is None:
        tensor = torch.zeros(1, 3, img_height, img_width)
        ratio = (1, 1)
        pad = (0.0, 0.0)
    else:
        img, ratio, pad = letterbox(
            img,
            new_shape=(img_height, img_width),
            auto=False,
        )
        tensor = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        tensor = np.ascontiguousarray(tensor)

        tensor = torch.from_numpy(tensor).to(device, non_blocking=True)
        tensor = tensor.float()
        tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)
    return tensor, ratio, pad

