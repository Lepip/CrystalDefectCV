import numpy as np

from .modules_wrapper import register_module
from .config import config as all_config
import cv2


def scharr_technique(img) -> np.ndarray:
    config = all_config["scharr_technique"]
    img_dxy = scharr_xy(img)
    log_img = np.sign(img_dxy) * np.log1p(np.abs(img_dxy))
    what = np.abs(log_img) > config["threshold"]
    blured = cv2.blur(src=what.astype(np.float64) * 255, ksize=(config["blur_size"], config["blur_size"]))
    blured = blured ** config["power"]
    blured = cv2.normalize(blured, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    blured[blured < np.mean(blured) + config["blur_threshold"]] = 0
    return blured


if all_config["use_scharr"]:
    scharr_technique = register_module(scharr_technique)


def scharr_xy(img):
    dx = cv2.Scharr(img, ddepth=cv2.CV_64F, dx=1, dy=0)
    dy = cv2.Scharr(img, ddepth=cv2.CV_64F, dx=0, dy=1)
    dxy = np.sqrt(dx*dx + dy*dy)
    return dxy
