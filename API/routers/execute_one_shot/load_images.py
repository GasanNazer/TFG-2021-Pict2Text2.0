import os
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.utils import to_categorical
from numpy import argmax

def load_images(folder = "pictograms", classes_loaded = 0):
    images = []
    Y = []
    folders = dict()
    class_index = 0
    for subfolder in os.listdir(folder):
        subfolder_complete_path = os.path.join(folder, subfolder)
        for subsubfolder in os.listdir(subfolder_complete_path):
            if subsubfolder not in folders:
                folders[subsubfolder] = to_categorical(class_index) # + classes_loaded)
                class_index += 1
            subsubfolder_complete_path = os.path.join(subfolder_complete_path, subsubfolder)
            folder_images = []
            for filename in os.listdir(subsubfolder_complete_path):
                # load the image
                img = load_img(os.path.join(subsubfolder_complete_path, filename), color_mode="rgba", target_size=(105, 105))

                x = img_to_array(img)
                folder_images.append(x)
                Y.append(folders[subsubfolder])
            images.append(np.array(folder_images))
    return np.array(images), np.reshape(np.array(Y), (-1, 1)), {argmax(v): k for k, v in folders.items()}

