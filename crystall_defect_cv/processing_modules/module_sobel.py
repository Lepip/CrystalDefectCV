import numpy as np

from .modules_wrapper import register_module
from .config import config as all_config
import cv2


def sobel_technique(img) -> np.ndarray:
    config = all_config["sobel_technique"]
    if config["ksize"] % 2 == 0:
        config["ksize"] += 1
    img_sobeledxy = sobel_xy(img, int(config["ksize"]))
    log_img = np.sign(img_sobeledxy) * np.log1p(np.abs(img_sobeledxy))
    what = np.abs(log_img) > config["threshold"]
    blured = cv2.blur(src=what.astype(np.float64) * 255, ksize=(config["blur_size"], config["blur_size"]))
    blured = blured ** config["power"]
    blured = cv2.normalize(blured, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    blured[blured < np.mean(blured) + config["blur_threshold"]] = 0
    return blured


if all_config["use_sobel"]:
    sobel_technique = register_module(sobel_technique)


def sobel_xy(img, ksize):
    return cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=ksize)
