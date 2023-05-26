import numpy as np

from .modules_wrapper import register_module
from .config import config as all_config
import cv2


@register_module
def scharr_technique(img) -> np.ndarray:
    config = all_config["scharr_technique"]
    img_dxy = scharr_xy(img)
    return img_dxy


def scharr_xy(img):
    dx = cv2.Scharr(img, ddepth=cv2.CV_64F, dx=1, dy=0)
    dy = cv2.Scharr(img, ddepth=cv2.CV_64F, dx=0, dy=1)
    dxy = np.sqrt(dx*dx + dy*dy)
    return dxy
