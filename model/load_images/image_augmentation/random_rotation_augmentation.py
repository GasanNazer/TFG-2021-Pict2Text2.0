from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator

def create_random_rotation_augmentation_iterator(image_path):
    img = load_img(image_path, color_mode="rgba")
    samples = expand_dims(img, 0)
    datagen = ImageDataGenerator(rotation_range=90)
    it = datagen.flow(samples, batch_size=1)
    return it
