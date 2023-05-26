import cv2
import numpy as np
import os
import crystall_defect_cv as cdcv
import random
from PIL import Image

image_dirs = os.listdir("./pictures")
size_img = 100
num_red = 40321
num_plain = 35987
step = 5
for image_dir in image_dirs:
    if image_dir.find("red") != -1:
        img = cv2.imread("./pictures/" + image_dir)
        name = image_dir.split("_")
        img_orig = cv2.imread("./pictures/" + name[0] + ".png")
        shape = img.shape
        for x in range(0, shape[0] - size_img - step - 1, step):
            for y in range(0, shape[1] - size_img - step - 1, step):
                img_part = img[x:x+size_img, y:y+size_img]
                itsred = 0
                if np.max(img_part, axis=(0, 1))[2] > np.max(img_part, axis=(0, 1))[1]:
                    itsred = 1
                if (itsred == 1) and (random.randint(1, 4) == 3):
                    img_orig_crop = img_orig[x:x+size_img, y:y+size_img]
                    if cv2.imwrite("./train/red/" + str(num_red) + ".png", img_orig_crop) == False:
                        print("aboba")
                    num_red += 1
                elif (itsred == 0) and (random.randint(1, 30) == 5):
                    img_orig_crop = img_orig[x:x + size_img, y:y + size_img]
                    cv2.imwrite("./train/nor/" + str(num_plain) + ".png", img_orig_crop)
                    num_plain += 1
        print("next image!", num_red, num_plain)

image_dirs = os.listdir("./train")
for dirr in image_dirs:
    for image_dir in os.listdir("./train/" + dirr):
        if image_dir.find("r") == -1:
            continue
        img = cv2.imread("./train/" + dirr + "/" + image_dir)
        img_rot = np.rot90(img)
        name = image_dir.split('.')
        cv2.imwrite("./train/" + dirr + "/" + name[0] + 'r.' + name[1], img_rot)
