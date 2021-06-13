from siamese_network import get_siamese_model
from keras.optimizers import Adam
from progressivly_load_images import test_pictogram_against_all, test_small_it
from load_images import load_images
import os

from keras_preprocessing.image import ImageDataGenerator, array_to_img, load_img, img_to_array


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model = get_siamese_model((105, 105, 4))
    optimizer = Adam(lr=0.00006)
    model.compile(loss="binary_crossentropy", optimizer=optimizer)
    evaluate_every = 1  # interval for evaluating on one-shot tasks
    batch_size = 20
    n_iter = 50 # No. of training iterations
    N_way = 5  # how many classes for testing one-shot tasks
    n_val = 5  # how many one-shot tasks to validate on
    best = -1
    model_path = './weights'

    TARGET_SIZE = (105, 105)
    #photo = "test25.png"
    batchX_test, batchY_test, folders = load_images("pictograms_test_small")
    weights = [950, 1550, 1850]
    for i in weights:
        print("weigth: " + str(i))
        model.load_weights(os.path.join('./weights', 'weights.' + str(i) + '.h5'))
        val_acc = 0
        for photo in os.listdir("./test"):
            img = img_to_array(load_img(os.path.join("./test", photo), color_mode="rgba", target_size=TARGET_SIZE))
            id, predicted_pictogram = test_pictogram_against_all(model, batchX_test[:, 0, :, :, :], img, folders)
            print("weights: " + str(i))
            print(id)
            print(photo)
            if str(id) in photo:
                val_acc += 1
                print(val_acc)
        print("accuracy: ")
        val_acc /= 20
        print(val_acc)

    model.summary()
