from ..execute_one_shot.siamese_network import get_siamese_model
from keras.optimizers import Adam
from ..execute_one_shot.load_images import load_images
import numpy as np
import os

from keras_preprocessing.image import load_img, img_to_array

def test_pictogram_against_all(model, X, pictogram, folders):
    n_classes, w, h, d = X.shape
    test_image = np.asarray([pictogram] * n_classes)
    test_image = test_image.reshape(n_classes, w, h, d)
    pairs = [test_image, X]
    probs = model.predict(pairs)
    predicted = np.argmax(probs)

    return folders[predicted], X[predicted], probs[predicted]

def make_prediction_one_shot(pictograms_folder = './convert_jpg_to_png/png_images', test_pictograms = "pictograms", weights_folder = './weights'):
    ## One-Shot test configuration

    model = get_siamese_model((105, 105, 4))
    optimizer = Adam(lr=0.00006)
    model.compile(loss="binary_crossentropy", optimizer=optimizer)

    TARGET_SIZE = (105, 105)
    # directory with pictograms to compare against
    batchX_test, batchY_test, folders = load_images(test_pictograms)

    weight_path = weights_folder
    weight = 1850 # best weights
    model.load_weights(os.path.join(weight_path, f'weights.{weight}.h5'))

    ## One-Shot test configuration end

    # directory with cropped images which the model will use to classify
    dir_cropped_images = pictograms_folder

    classifications = []
    for photo in os.listdir(dir_cropped_images):
        img = img_to_array(
            load_img(os.path.join(dir_cropped_images, photo), color_mode="rgba", target_size=TARGET_SIZE))
        id, predicted_pictogram, probability = test_pictogram_against_all(model, batchX_test[:, 0, :, :, :], img,
                                                                          folders)
        print(f"Tested image with name {photo}. Predicted as: {id} with similarity {probability}")

        classifications.append({"id": id, "similarity": probability[0]})


    return classifications


#print(make_prediction_one_shot())