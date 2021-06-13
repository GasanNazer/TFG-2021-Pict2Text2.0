from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

def create_random_brightness_augmentation_iterator(image_path):
    img = load_img(image_path, color_mode="rgba")
    # convert to numpy array
    data = img_to_array(img)
    # expand dimension to one sample
    samples = expand_dims(data, 0)
    # create image data augmentation generator
    datagen = ImageDataGenerator(brightness_range=[0.2, 1.0])
    # prepare iterator
    it = datagen.flow(samples, batch_size=1)
    return it
