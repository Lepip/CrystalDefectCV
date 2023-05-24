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
