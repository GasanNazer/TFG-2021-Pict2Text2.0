import os
from keras.preprocessing.image import load_img, img_to_array, array_to_img
import tensorflow as tf
import numpy as np


def load_images():
    folder = "pictograms"
    images = []
    Y = []
    folders = dict()
    class_index = 0
    for subfolder in os.listdir(folder):
        subfolder_complete_path = os.path.join(folder, subfolder)
        for subsubfolder in os.listdir(subfolder_complete_path):
            if subsubfolder not in folders:
                folders[subsubfolder] = class_index
                class_index += 1
            subsubfolder_complete_path = os.path.join(subfolder_complete_path, subsubfolder)
            folder_images = []
            images_class = []
            for filename in os.listdir(subsubfolder_complete_path):
                # load the image
                img = load_img(os.path.join(subsubfolder_complete_path, filename), color_mode="rgba", target_size=(105, 105))
                # report details about the image
                #print(type(img))
                #print(img.format)
                #print(img.mode)
                #print(img.size)
                # show the image
                #img.show()
                x = img_to_array(img)
                folder_images.append(x)
                Y.append(folders[subsubfolder])
            images.append(np.array(folder_images))
        #return tf.keras.preprocessing.image_dataset_from_directory(directory = subfolder_complete_path,color_mode="rgba")
    return np.array(images), np.reshape(np.array(Y), (-1, 1)), {v: k for k, v in folders.items()}

def show_image_from_array(images_array, image):
    array_to_img(images_array[image][0]).show()

#X, Y, folders = load_images()

#show_image_from_array(X, 0)
