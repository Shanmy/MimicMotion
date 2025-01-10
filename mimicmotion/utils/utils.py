import logging
from pathlib import Path
import numpy as np

from torchvision.io import write_video
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

from lang_sam import LangSAM

langsam_model = LangSAM()

def save_to_mp4(frames, save_path, fps=7):
    if frames.shape[1] == 3:
        frames = frames.permute((0, 2, 3, 1))  # (f, c, h, w) to (f, h, w, c)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    write_video(save_path, frames, fps=fps)


def paste_crop(frames, bg, pix_offset):
    # frames: animated human frame
    # bg: background
    # pix_offset: (nframe * 4). a[0], a[1], b-offset[0], b-offset[1]
    crop_width = int(1 / pix_offset[0][0] * bg.size[0])
    crop_height = int(1 / pix_offset[0][1] * bg.size[1])
    top_left_w = (- pix_offset[:, 2] / pix_offset[0, 0] * bg.size[0]).astype(int)
    top_left_h = (- pix_offset[:, 3] / pix_offset[0, 1] * bg.size[1]).astype(int)
    comp_frame = []
    resize_crop = F.interpolate(frames, size=(crop_height, crop_width), mode='bilinear', align_corners=False)
    for i in range(frames.shape[0]):
        comp_frame.append(np.array(bg))
        #masks = langsam_model.predict([Image.fromarray(resize_crop[i])], ["human"])
        comp_frame[i][top_left_h[i]:top_left_h[i] + crop_height, top_left_w[i]:top_left_w[i] + crop_width] = resize_crop[i].permute(1, 2, 0)
    return np.array(comp_frame).astype(np.uint8)
