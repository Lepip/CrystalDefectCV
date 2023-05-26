import os

import numpy as np
from PIL import Image
from PIL import ImageChops
from termcolor import colored

from . import utils
from .processing_modules import modules_dict


def find_defects_probs(image: np.ndarray) -> np.ndarray:
    """
    Processes the image and returns a matrix of defect certainty.
    :param image: Gray image
    :returns: Matrix of defect certainties
    """
    probability_matrices = {module_name: module(image) for module_name, module in modules_dict.items()}
    output_probs = merge_probability_matrices(probability_matrices)
    return output_probs


def mark_defects(image: np.ndarray, defects_matrix: np.ndarray, min_confidence_threshold: float = 0.5):
    """
    Marks defects on the image based on the probability matrix.
    """
    from .utils import is_greater
    marked_matrix = is_greater(defects_matrix, min_confidence_threshold)
    marked_image = Image.fromarray(image, mode="L").convert("RGB")
    red_color = Image.new("RGB", image.size, color=(255, 0, 0))
    marked_image = ImageChops.composite(marked_image, red_color, mask=marked_matrix)
    return marked_image


def merge_probability_matrices(probability_matrices: {str: np.ndarray}) -> np.ndarray:
    """
    Merges
    :param probability_matrices: Array of probability matrices produced by different defect detecting techniques.
    :returns: Matrix of merged probability matrices into one.
    """
    if len(probability_matrices) == 0:
        print(colored("Error: no module output presented for merge_probability_matrices", color="red"))
        return np.zeros([1, 1])
    print(probability_matrices)
    output_matrix = np.zeros(probability_matrices[0].size)
    for probability_matrix in probability_matrices:
        output_matrix = np.maximum(output_matrix, probability_matrix)
    return output_matrix


def evaluate_efficiency(params):
    import cv2
    from . import io
    effectiveness = 0
    losses = 0
    ok = 0
    not_ok_num = 0
    global_not_ok = 0
    first = 1
    num_img = 0
    images_dirs = os.listdir("./pictures")
    for image_dir in images_dirs:
        if image_dir.find("_") != -1:
            continue
        name = image_dir.split('.')
        img = io.open_png("./pictures/" + image_dir)
        img_xy = params["operator"](img)
        log_img = np.sign(img_xy) * np.log1p(np.abs(img_xy))
        what = np.abs(log_img) > params["threshold"]
        blured = cv2.blur(src=what.astype(np.float64) * 255, ksize=(34, 34)) ** params["power"]
        blured = cv2.normalize(blured, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        blured[blured < np.mean(blured) + 20] = 0
        contours, hierarchy = cv2.findContours(blured, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        red_image = cv2.imread("./pictures/" + name[0] + "_red.png")
        not_ok = 0
        img10 = red_image*10
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            w = h = max(w, h)
            if params["show"] == 1:
                cv2.rectangle(img10, (x, y), (x + w, y + h), (0, 0, 255), 2)
            red_crop = red_image[x:x+w, y:y+h]
            mean = np.mean(red_crop, axis=(0, 1))
            if mean[2] > mean[1]:
                ok += 1
            else:
                not_ok += w*h
                not_ok_num += 1
        global_not_ok += not_ok/1500./1500.
        num_img += 1
        if params["show"] == 1:
            cv2.imshow(name[0], img10)
    global_not_ok /= num_img
    if params["show"] == 1:
        cv2.waitKey(0)
    return ok / params["hm"], not_ok_num / params["hm"], global_not_ok
