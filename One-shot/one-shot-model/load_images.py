import os

from PIL.Image import Image
from keras.preprocessing.image import load_img, img_to_array, array_to_img
import tensorflow as tf
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
    a = np.array(images)
        #return tf.keras.preprocessing.image_dataset_from_directory(directory = subfolder_complete_path,color_mode="rgba")
    return np.array(images), np.reshape(np.array(Y), (-1, 1)), {argmax(v): k for k, v in folders.items()}

def load_test_images(folder = "pictograms_test", MAX = 200):
    images = []
    folders = dict()
    images_id = 0
    for filename in os.listdir(folder):
        img = load_img(os.path.join(folder, filename), color_mode="rgba", target_size=(105, 105))
        x = img_to_array(img)

        image_id = filename.split("-")[0]
        folders[images_id] = image_id

        images_id += 1
        images.append(x)

        if MAX <= images_id:
            break
        #return tf.keras.preprocessing.image_dataset_from_directory(directory = subfolder_complete_path,color_mode="rgba")
    return np.array(images), folders


def show_image_from_array(images_array, image):
    array_to_img(images_array[image][0]).show()

#X, Y, folders = load_images()

#X_val, Y_val, folders_val = load_images("pictograms_val", classes_loaded= len(folders))

#show_image_from_array(X, 0)
#show_image_from_array(X, 1)

def show_two_images(image_array1, image1, image_array2, image2):
    a = array_to_img(image_array1[image1][0])
    b = array_to_img(image_array2[image2][0])

    import matplotlib.pyplot as plt

    plt.imshow(np.asarray(a))

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(a)
    fig.add_subplot(1, 2, 2)
    plt.imshow(b)
    plt.show()

#show_two_images(X, 0, X, 1)


#X_test, folders = load_test_images(MAX= 1000)