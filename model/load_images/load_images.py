import os
from keras.preprocessing.image import load_img, img_to_array, array_to_img

def load_images():
    folder = "pictograms"
    images = []
    for subfolder in os.listdir(folder):
        subfolder_complete_path = os.path.join(folder, subfolder)
        for filename in os.listdir(subfolder_complete_path):
            # load the image
            img = load_img(os.path.join(subfolder_complete_path, filename), color_mode="rgba")
            # report details about the image
            #print(type(img))
            #print(img.format)
            #print(img.mode)
            #print(img.size)
            # show the image
            #img.show()
            x = img_to_array(img)
            images.append(x)
    return images

def show_image_from_array(images_array, image):
    array_to_img(images_array[image]).show()


images = load_images()
show_image_from_array(images, 16)