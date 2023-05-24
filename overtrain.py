import pathlib

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds

batch_size = 32
img_height = 100
img_width = 100
data_dir = "./train"
data_dir = pathlib.Path(data_dir)
print(len(list(data_dir.glob('*/*.png'))))

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

model = tf.keras.models.load_model("./models/model1.ckpt")

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=10
)

model.save("./models/model2.ckpt")
