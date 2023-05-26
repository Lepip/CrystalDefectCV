import numpy as np

from .modules_wrapper import register_module
from .config import config as all_config
import cv2
from matplotlib import pyplot as plt


# @register_module
def sobel_technique(img) -> np.ndarray:
    config = all_config["sobel_technique"]
    img_sobeledxy = sobel_xy(img, 1 + int(config["defect_size"]) * 2)
    return img_sobeledxy
    # return blured


def sobel_xy(img, ksize):
    return cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=ksize)
