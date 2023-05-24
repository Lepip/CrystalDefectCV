from PIL import Image
import numpy as np


def open_png(file_path: str) -> np.ndarray:
    """
    Reads a PNG file from 'file_path', grayscales it and returns 2D ndarray of uint8.
    :param file_path: Path of the PNG file.
    :return: 2D ndarray, dtype='uint8'
    :rtype: np.ndarray
    """
    image = Image.open(file_path)
    image_gray = image.convert("L")
    image_array = np.array(image_gray.getdata()).astype('uint8').reshape(image_gray.size)
    return image_array


def save_png(image: np.ndarray, file_path: str) -> None:
    """
    Writes a 2D ndarray of uint8 as a grey image.
    :param file_path:
    :param image: gray image.
    :type image: 2D np.ndarray, dtype='uint8'
    :return: None
    """
    image_gray = Image.fromarray(image, mode="L")
    image_gray.save(file_path)
