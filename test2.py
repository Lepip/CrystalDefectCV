import numpy as np

import crystall_defect_cv as cdcv
from crystall_defect_cv.processing_modules import module_sobel
import cv2
# image = cdcv.open_png("pictures/006850.png")

# defects_matrix = cdcv.find_defects_probs(image)

params = {
    "operator": cdcv.processing_modules.scharr_technique,
    "threshold": 3.4,
    "power": 3.0,
    "hm": 130,
    "show": 1
}

cdcv.evaluate_efficiency(params)

#
# for threshold in range(20, 50, 3):
#     threshold_f = threshold / 10
#     for power in range(30, 10, -5):
#         power_f = power/10
#         params["threshold"] = threshold_f
#         params["power"] = power_f
#         outputs = cdcv.evaluate_efficiency(params)
#         print(threshold_f, power_f, outputs)
#         if outputs[1] > 10:
#             break

# defects_image = cdcv.mark_defects(image, defects_matrix, min_confidence_threshold=0.3)sub

# cdcv.save_png(defects_image, "test_processed.png")

# import tensorflow as tf

# def draw_circle(event,x,y,flags,param):
#     if(event == cv2.EVENT_LBUTTONUP):
#         img_part = picture[x - size_img//2:x + size_img//2, y-size_img//2:y + size_img//2]
#         print(model.predict(np.expand_dims(img_part, axis=0), verbose=0))
#
#
# cv2.namedWindow('img')
# cv2.setMouseCallback('img', draw_circle)
# model = tf.keras.models.load_model("./models/model1.ckpt")
# print(model.summary())
#
# picture = cv2.imread("./pictures/006450.png")
#
# img = picture.copy()
# img *= 10
# shape = picture.shape
# size_img = 100
# step = 50
# # for x in range(0, shape[0] - size_img - step - 1, step):
# #     for y in range(0, shape[1] - size_img - step - 1, step):
# #         img_part = picture[x:x + size_img, y:y + size_img]
# #         prediction = model.predict(np.expand_dims(img_part, axis=0), verbose=0)
# #         if prediction[0][0] > prediction[0][1]:
# #             cv2.rectangle(img, (x - size_img//2, y - size_img//2),
# #                       (x + size_img//2, y + size_img//2),
# #                       (0, 0, 255), 3)
# #     print(x)
# cv2.imshow("img", img)
# cv2.waitKey(0)
