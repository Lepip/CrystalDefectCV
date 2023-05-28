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
    Writes a ndarray of uint8 as an image file. If 'image' has only 2 dimensions, writes it as a gray image.
    :param file_path:
    :param image: gray image.
    :type image: 2D or 3D (*, *, 3) np.ndarray, dtype='uint8'
    :return: None
    """
    if len(image.shape) == 2:
        image_gray = Image.fromarray(image.astype('uint8'), mode="L")
        image_gray.save(file_path)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = Image.fromarray(image.astype('uint8'), mode="RGB")
        image_rgb.save(file_path)
    else:
        print("Error in cdcv.save_png: expected shape of image to be (*, *) or (*, *, 3), got " + str(image.shape) + ".")
