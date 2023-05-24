import math
import time

from matplotlib import pyplot as plt

import crystall_defect_cv as cdcv
import cv2
import numpy as np


def sobel_technique(img) -> np.ndarray:
    img_sobeledxy = np.float32(sobel_xy(img, 1 + 6 * 2))
    # blured = cv2.blur(img_sobeledxy, (10, 10))
    blured = img_sobeledxy
    blured = cv2.normalize(src=blured, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # cv2.imshow("blured", blured.astype(np.uint8))
    # plt.figure(1)
    # plt.imshow(cv2.cvtColor(blured, cv2.COLOR_BGR2RGB))
    # plt.show()
    # print(blured.dtype, np.max(blured))
    #heatmap = cv2.applyColorMap(blured, cv2.COLORMAP_RAINBOW)
    #cv2.imshow("heatmap", heatmap)
    return blured.astype(np.uint8)


def sobel_xy(img, ksize):
    # return cv2.Sobel(src=img, ddepth=cv2.CV_8U, dx=1, dy=1, ksize=ksize)
    ox = cv2.Sobel(src=img, ddepth=cv2.CV_8U, dx=1, dy=0, ksize=ksize)
    # cv2.imshow("ox", cv2.Sobel(src=img, ddepth=cv2.CV_8U, dx=1, dy=0, ksize=ksize))
    # cv2.imshow("oy", cv2.Sobel(src=img, ddepth=cv2.CV_8U, dx=0, dy=1, ksize=ksize))
    oy = cv2.Sobel(src=img, ddepth=cv2.CV_8U, dx=0, dy=1, ksize=ksize)
    return cv2.addWeighted(ox, 0.5, oy, 0.5, 0)


def get_image(img, angle):
    h, w = img.shape[:2]
    new_img = rotation(img, angle)
    sob = sobel_technique(new_img)
    sob_rot = rotation(sob, -angle)
    h2, w2 = sob_rot.shape[:2]
    funny = (h2-h)//2
    ret_img = sob_rot[funny:h+funny, funny:w+funny]
    return ret_img
lol_image = cv2.imread("test.png")

def rotation(image, angleInDegrees):
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(img_c, angleInDegrees, 1)

    rad = math.radians(angleInDegrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
    return outImg


cv2.imshow("abobalol", lol_image*15)
# lol_image = cv2.blur(lol_image, (3, 3))

while True:
    cv2.waitKey(0)
im_fft = np.fft.fft2(cv2.cvtColor(lol_image, cv2.COLOR_BGR2GRAY))
keep_fraction = 0.3
delete_fraction = 0
im_fft2 = im_fft.copy()
r, c = im_fft2.shape
im_fft2[int(r*keep_fraction):r] = 0
im_fft2[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0
lol_image = np.array(np.fft.ifft2(im_fft2).real, dtype="uint8")
cv2.imshow("lol", lol_image*15)
def_sobel = sobel_technique(lol_image)
def_sobel = def_sobel.astype(np.uint64)
cnt = 1
mx = 179
cv2.imshow("funny", get_image(lol_image, 180))
cv2.imshow("funny2", def_sobel.astype(np.uint8))
for i in range(1, mx):
    funny_image = get_image(lol_image, (i))
    def_sobel += funny_image.astype(np.uint64)
    cnt += 1
def_sobel //= cnt
def_sobel = def_sobel.astype(np.uint8)
cv2.imshow("aboba", def_sobel)
plt.figure(1)
plt.imshow(cv2.cvtColor(def_sobel, cv2.COLOR_BGR2RGB))
plt.show()
cv2.waitKey(0)
# img = cv2.cvtColor(lol_image, cv2.COLOR_BGR2GRAY)
# kernel = np.ones((1, 1), np.float32)*25
# dst = cv2.filter2D(img, -1, kernel)
# median = cv2.medianBlur(dst, 3)
#
# im_fft = np.fft.fft2(dst)
# keep_fraction = 1
# delete_fraction = 0
# im_fft2 = im_fft.copy()
# r, c = im_fft2.shape
# im_fft2[int(r*keep_fraction):r] = 0
# im_fft2[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0
# im_fft2[0:int(r*delete_fraction)] = 0
# im_fft2[:, 0:int(r*delete_fraction)] = 0
# im_new = np.array(np.fft.ifft2(im_fft2).real, dtype="uint8")
# current_thing = 1
#
#
#
# dst = cv2.blur(dst, (5, 5))
# cv2.imshow("new", im_new)
# cv2.imshow("old", dst)
sobel_technique(lol_image, 10000)
def on_change(value):
    sobel_technique(lol_image, value)
    # kernel = np.ones((1, 1), np.float32) * value/100
    # global current_thing
    # current_thing = value
    # cv2.imshow("old", dst-cv2.filter2D(im_new, -1, kernel))
    # cv2.imshow("new", cv2.filter2D(im_new, -1, kernel))

# def on_fft(value):
#     keep_fraction = value/1000
#     kernel = np.ones((1, 1), np.float32) * value/10
#     im_fft2 = im_fft.copy()
#     r, c = im_fft2.shape
#     im_fft2[int(r * keep_fraction):int(r * (1 - keep_fraction))] = 0
#     im_fft2[:, int(c * keep_fraction):int(c * (1 - keep_fraction))] = 0
#     global im_new
#     im_new = np.array(np.fft.ifft2(im_fft2).real, dtype="uint8")
#     on_change(current_thing)
cv2.createTrackbar('blured', "blured", 0, 400, on_change)
#cv2.createTrackbar('fft', 'old', 0, 1000, on_fft)
while True:
    cv2.waitKey(0)


image = cdcv.open_png("test.png")

defects_matrix = cdcv.find_defects_probs(image)

defects_image = cdcv.mark_defects(image, defects_matrix, min_confidence_threshold=0.3)

cdcv.save_png(defects_image, "test_processed.png")
