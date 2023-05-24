import numpy as np

from .modules_wrapper import register_module
from .config import config as all_config
import cv2
from matplotlib import pyplot as plt


@register_module
def sobel_technique(img) -> np.ndarray:
    config = all_config["sobel_technique"]
    # cv2.imshow("img", img*10)
    img_sobeledxy = sobel_xy(img, 1 + int(config["defect_size"]) * 2)
    fig, axs = plt.subplots(tight_layout=True)
    funny_array = np.sort(img_sobeledxy.flatten())
    nbins = 100
    bins = [0 for binid in range(nbins)]
    funny_array = np.sign(funny_array) * np.log1p(np.abs(funny_array))
    left = funny_array[0]
    print(funny_array[0])
    print(funny_array[-1])
    step = (funny_array[-1] - funny_array[0]) / nbins
    for binid in range(nbins):
        right = left + step
        bins[binid] = np.count_nonzero((left <= funny_array) & (funny_array <= right))
        left += step
    print(bins)
    plt.yscale("log")
    axs.hist(funny_array, bins=1000)
    plt.show()
    log_img = np.sign(img_sobeledxy) * np.log1p(np.abs(img_sobeledxy))
    cv2.imshow("img", img*10)
    while (True):
        inp = input()
        what = eval(inp, {"x": log_img, "np": np})
        img_disp = what.astype("uint8")*255
        cv2.imshow("disp", img_disp)
        blured = cv2.blur(src=what.astype(np.float64)*255, ksize=(34, 34))**5
        blured = cv2.normalize(blured, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imshow("blured", blured)
        cv2.waitKey(0)
        cv2.destroyWindow("disp")
        cv2.destroyWindow("blured")
    # blured = cv2.blur(img_sobeledxy, (10, 10))/187872470.
    # cv2.imshow("blured", blured.astype(np.uint8))
    # print(blured.dtype, np.max(blured))
    #heatmap = cv2.applyColorMap(blured, cv2.COLORMAP_RAINBOW)
    #cv2.imshow("heatmap", heatmap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # return blured


def sobel_xy(img, ksize):
    return cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=ksize)
