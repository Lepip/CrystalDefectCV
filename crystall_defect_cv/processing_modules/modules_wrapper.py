modules_dict = {}


def register_module(func):
    """
    Decorator for modules that process images into probability matrices.
    :param func:
    :return:
    """
    modules_dict[func.__name__] = func
    return func


__all__ = ["modules_dict", "register_module"]
