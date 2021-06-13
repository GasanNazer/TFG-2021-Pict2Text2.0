from keras.preprocessing.image import save_img
import os

from random_brightness_augmentation import create_random_brightness_augmentation_iterator
from random_rotation_augmentation import create_random_rotation_augmentation_iterator
from color_augmentation import generate_images_with_augmented_color

def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def iterate_and_save_augmented_images(iterator, image_folder, filename, augmentation_type, number_of_samples):
    count = 1
    for i in iterator:
        save_img(os.path.join(image_folder, f'{filename.split(".")[0]}_{augmentation_type}_{count}.png'), i[0], file_format="png")
        count += 1
        if count > number_of_samples:
            break

def augment_images_from_folder(folder = "../pictograms", file_suffix = ""):
    for subfolder in os.listdir(folder):
        subfolder_complete_path = os.path.join(folder, subfolder)
        for filename in os.listdir(subfolder_complete_path):
            if os.path.isfile(os.path.join(subfolder_complete_path, filename)):
                image_folder = os.path.join(subfolder_complete_path, filename.split("-")[0])
                make_directory(image_folder)

                original_filename = filename
                filename = os.path.splitext(filename)[0] + "_" + file_suffix + ".png"

                it_brightness = create_random_brightness_augmentation_iterator(os.path.join(subfolder_complete_path, original_filename))
                it_rotation = create_random_rotation_augmentation_iterator(os.path.join(subfolder_complete_path, original_filename))

                iterate_and_save_augmented_images(it_brightness, image_folder, filename, "brightness", 10)
                iterate_and_save_augmented_images(it_rotation, image_folder, filename, "rotation", 4)

                generate_images_with_augmented_color(os.path.join(subfolder_complete_path, original_filename), filename, image_folder, "color", 6)

                os.rename(os.path.join(subfolder_complete_path, original_filename), os.path.join(image_folder, original_filename))

    print("Augmentation finished!")

augment_images_from_folder("/media/roni/External_PRO/TFG/pictograms/project/model/load_images/convert_jpg_to_png", "photo")
