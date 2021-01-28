import cv2
import numpy as np
from PIL import Image
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array, save_img
from keras.preprocessing.image import ImageDataGenerator
import os

filename = "2239-abeja.png"
image = load_img(os.path.join("../pictograms/0/", filename), color_mode="rgba")

train_datagen = ImageDataGenerator(channel_shift_range=10, rescale=1./255)
samples = expand_dims(image, 0)
it = train_datagen.flow(samples, batch_size=1)

count = 0

for i in it:
    print(i)
    #show_image_from_array(i, 0)
    save_img(f'a_{count}.png', i[0], file_format="png")
    count += 1

    if count > 5:
        break