import numpy as np

MAX_CONFIDENCE = 255


def sigmoid(x):
    """
    Возвращает сигмоидную функцию от входного массива.
    """
    return 1 / (1 + np.exp(-x))


def is_greater_pyfunc(value, threshold):
    if value >= threshold:
        return MAX_CONFIDENCE
    else:
        return 0


is_greater = np.vectorize(is_greater_pyfunc)
