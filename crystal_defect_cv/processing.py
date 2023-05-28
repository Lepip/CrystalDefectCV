import os

import numpy as np
import cv2

from .processing_modules import modules_dict


def find_defects_probs(image: np.ndarray) -> np.ndarray:
    """
    Processes the image and returns a matrix of defect certainty.
    :param image: Gray image.
    :returns: Matrix of defect certainties.
    """
    probability_matrices = {module_name: module(image) for module_name, module in modules_dict.items()}
    output_probs = merge_probability_matrices(probability_matrices)
    return output_probs


def mark_defects(image: np.ndarray, defects_matrix: np.ndarray, lighten_image_up=True):
    """
    Marks defects on the image based on the probability matrix.
    :param image: An image to mark defects on.
    :param defects_matrix: Matrix of the same size as image, every nonzero element is treated like a defect.
    :param lighten_image_up: If True, lights up the image by 10.
    :returns: Marked RGB image with red squares.
    """
    marked_image = image.copy()
    marked_image = cv2.cvtColor(marked_image * (lighten_image_up * 9 + 1), cv2.COLOR_GRAY2RGB)
    contours, hierarchy = cv2.findContours(defects_matrix, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        w = h = max(w, h)
        cv2.rectangle(marked_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return marked_image


def merge_probability_matrices(probability_matrices: {str: np.ndarray}) -> np.ndarray:
    """
    Merges matrices. Returns element-wise max of them.
    :param probability_matrices: Array of probability matrices produced by different defect detecting techniques.
    :returns: Matrix of merged probability matrices into one.
    """
    if len(probability_matrices) == 0:
        print("Error: no module output presented for merge_probability_matrices")
        return np.zeros([1, 1])
    shape = list(probability_matrices.values())[0].shape
    output_matrix = np.zeros(shape).astype('uint8')
    for probability_matrix in probability_matrices.values():
        output_matrix = np.maximum(output_matrix, probability_matrix)
    return output_matrix


def mark_all_in_directory(operator, directory, save_directory):
    """
    Marks all the images in the directory, using the operator as defect-detect technique and saves results
    in the save_directory.
    :param save_directory: directory to save marked images to.
    :param operator: function (img)->img, that marks the defects. For example "sobel_technique" from module_sobel.py.
    :param directory: directory of .png images with defects.
    """
    import cv2
    from . import io
    images_dirs = os.listdir(directory)
    for image_name in images_dirs:
        if image_name.find("_") != -1:
            continue
        img = io.open_png(directory + "/" + image_name)
        img_xy = operator(img)
        img_razm = mark_defects(img, img_xy)
        cv2.imwrite(save_directory + "/" + image_name, img_razm)
    cv2.waitKey(0)
