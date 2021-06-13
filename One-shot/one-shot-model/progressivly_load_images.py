from keras_preprocessing.image import ImageDataGenerator, array_to_img, load_img, img_to_array

from siamese_network import get_siamese_model
#from one_shot import get_batch, test_oneshot, test
from keras.optimizers import Adam
import os
import numpy.random as rng
import numpy as np
from sklearn.utils import shuffle

import time

datagen = ImageDataGenerator()

NUMBER_OF_ITERATIONS = 39
# with 100 iterations => 6/7 of the testing set
# with 100 iterations not shuffled =>  0/7 of the testing set

# with 150 iterations => 7/7 of the testing set (105, 105) doesn't detect the one image
# with 150 iterations => 44/44 of the testing set (64, 64) doesn't detect the one image
# with 200 iterations => 44/44 of the testing set (64, 64) doesn't detect the one image
# with 300 iterations => 44/44 of the testing set (64, 64) doesn't detect the one image
# with 400 iterations => 44/44 of the testing set (64, 64) doesn't detect the one image
# with 400 iterations with 4 classes batch => 44/44 of the testing set (64, 64) doesn't detect the one image
# with 1000 iterations with 4 classes batch => 44/44 of the testing set (64, 64) doesn't detect the one image
# with 25 iterations with 4 classes batch  training set with photos and pictos mix => 44/44 of the testing set (64, 64) doesn't detect the sky and balls yes others no
# with 50 iterations with 2 classes batch  training set with photos and pictos mix => 44/44 of the testing set (64, 64) doesn't detect the pavement yes others no
# with 100 iterations with 2 classes batch training set with photos and pictos mix => only the balls were correctly detected
# with 150 iterations with 2 classes batch training set with photos and pictos mix => only the balls were correctly detected
CLASSES = 5
NUM_IMAGES_PER_CLASS = 40 # used when constructing batches with pairs
BATCH_SIZE = CLASSES * NUM_IMAGES_PER_CLASS
TARGET_SIZE = (105, 105)

# load and iterate training dataset
train_it = datagen.flow_from_directory('/media/roni/External_PRO/TFG/pictograms/png_new_pictograms/0', target_size= TARGET_SIZE, color_mode="rgba", class_mode='categorical', batch_size=BATCH_SIZE, shuffle= False)
# load and iterate validation dataset
val_it = datagen.flow_from_directory('pictograms_val', target_size= TARGET_SIZE, color_mode="rgba", class_mode='categorical', batch_size=BATCH_SIZE, shuffle= False)
# load and iterate test dataset
test_it = datagen.flow_from_directory('pictograms_test', target_size= TARGET_SIZE, color_mode="rgba", class_mode='categorical', batch_size=BATCH_SIZE, shuffle= False)

test_real_it = datagen.flow_from_directory('pictograms_test_real', target_size= TARGET_SIZE, color_mode="rgba", class_mode='categorical', batch_size=BATCH_SIZE, shuffle= False)

test_small_it = datagen.flow_from_directory('pictograms_test_small', target_size= TARGET_SIZE, color_mode="rgba", class_mode='categorical', batch_size=BATCH_SIZE, shuffle= False)

##############################################
model = get_siamese_model(train_it.image_shape)
optimizer = Adam(lr=0.00006)
model.compile(loss="binary_crossentropy", optimizer=optimizer)
evaluate_every = 1  # interval for evaluating on one-shot tasks
#batch_size = 20
n_iter = 10 # No. of training iterations
N_way = 5  # how many classes for testing one-shot tasks
best = -1
model_path = './weights/old_model'


def get_batch(batch_size, Xtrain, train_classes, s="train"):
    """
    Create batch of n pairs, half same class, half different class
    """

    # construct X in shape (num_classes, num_examples_per_class, w, h, d)

    num_classes = int(Xtrain.shape[0] / NUM_IMAGES_PER_CLASS)

    X = Xtrain.reshape(num_classes, NUM_IMAGES_PER_CLASS, Xtrain.shape[1], Xtrain.shape[2], Xtrain.shape[3])

    n_classes, n_examples, w, h, d = X.shape
    if n_classes < batch_size:
        batch_size = n_classes
    # randomly sample several classes to use in the batch
    categories = rng.choice(n_classes, size=(batch_size,), replace=False)

    # initialize 2 empty arrays for the input image batch
    pairs = [np.zeros((batch_size, h, w, d)) for i in range(2)]

    # initialize vector for the targets
    targets = np.zeros((batch_size,))

    # make one half of it '1's, so 2nd half of batch has same class
    targets[batch_size // 2:] = 1
    for i in range(batch_size):
        category = categories[i]
        idx_1 = 0
        pairs[0][i, :, :, :] = X[category, idx_1].reshape(w, h, d)
        idx_2 = rng.randint(1, n_examples)

        # pick images of same class for 1st half, different for 2nd
        if i >= batch_size // 2:
            category_2 = category
        else:
            # add a random number to the category modulo n classes to ensure 2nd image has a different category
            try:
                category_2 = (category + rng.randint(1, n_classes)) % n_classes
            except Exception:
                print("Not enough pictograms in the training set. Cannot construct pairs with categories.")


        pairs[1][i, :, :, :] = X[category_2, idx_2].reshape(w, h, d)

    return pairs, targets


def show_batch_images(batch):
    import matplotlib.pyplot as plt

    size = 10

    fig = plt.figure(figsize=(size, size))
    columns = size
    rows = size
    count = 0
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        img = batch[count]
        plt.imshow(array_to_img(img))

        count += 1

        if len(batch) <= count:
            break

    plt.show()


def make_oneshot_task(batch_size, Xtrain, train_classes, N, pictogram=None):
    """Create pairs of test image, support set for testing N way one-shot learning. """

    num_classes = int(Xtrain.shape[0] / NUM_IMAGES_PER_CLASS)
    X = Xtrain.reshape(num_classes, NUM_IMAGES_PER_CLASS, Xtrain.shape[1], Xtrain.shape[2], Xtrain.shape[3])
    n_classes, n_examples, w, h, d = X.shape

    if N > n_classes:
        N = n_classes

    indices = rng.randint(0, n_examples, size=(N,))
    categories = train_classes
    if pictogram is not None:  # if pictogram is specified
        low, high = categories[pictogram]
        if N > high - low:
            raise ValueError("This pictogram ({}) has less than {} letters".format(pictogram, N))
        categories = rng.choice(range(low, high), size=(N,), replace=False)

    else:  # if no pictogram specified just pick a bunch of random ones
        categories = rng.choice(range(n_classes), size=(N,), replace=False)

    true_category = categories[0]
    ex1, ex2 = rng.choice(n_examples, replace=False, size=(2,))
    test_image = np.asarray([X[true_category, ex1, :, :, :]] * N).reshape(N, w, h, d)
    support_set = X[categories, indices, :, :, :]
    support_set[0, :, :, :] = X[true_category, ex2]
    support_set = support_set.reshape(N, w, h, d)
    targets = np.zeros((N,))
    targets[0] = 1
    targets, test_image, support_set = shuffle(targets, test_image, support_set)
    pairs = [test_image, support_set]

    return pairs, targets

n_correct = 0
last_weights_file = ""

def train_model():
    global n_correct, last_weights_file

    best_loss = 10000
    '''
    for i in range(1, NUMBER_OF_ITERATIONS + 1):
        batchX, batchy = train_it.next()
        (inputs, targets) = get_batch(BATCH_SIZE, batchX, batchy)
        loss = model.train_on_batch(inputs, targets)
        print(f"Loss:{loss}")
    '''
    loss = model.fit(train_it, targets, nb_epoch=NUMBER_OF_ITERATIONS, batch_size=20, verbose=0, validation_split=0.1)
    if best_loss >= loss:
        best_loss = loss
        print(f"Best loss: {best_loss}")
        batchX_val, batchY_val = val_it.next()
        inputs, targets = make_oneshot_task(BATCH_SIZE, batchX_val, batchY_val, N_way)
        probs = model.predict(inputs)
        # print(f"Probabilities: {probs}")

        if np.argmax(probs) == np.argmax(targets):
            n_correct += 1

        #if i % 5 == 0:
        last_weights_file = f'weights.{i}.h5'
        model.save_weights(os.path.join(model_path, last_weights_file))

        print(f"Iteration number: {i}")

def test_one_pictogram(model, X, pictogram_num = 5):
    n_classes, w, h, d = X.shape
    test_image = np.asarray([X[pictogram_num]] * n_classes)
    test_image = test_image.reshape(n_classes, w, h, d)
    pairs = [test_image, X]
    #start_time = time.time()
    probs = model.predict(pairs)
    #print(f"Calculating probability in {(time.time() - start_time)} seconds.")
    #print("Probabilities: ")
    #print(probs)
    predicted = np.argmax(probs)
    print(f"Id real: {folders[pictogram_num]}")
    print(f"Id: {folders[predicted]}")
    print(probs[predicted])

    return folders[pictogram_num] == folders[predicted]


def test_pictogram_against_all(model, X, pictogram, folders):
    n_classes, w, h, d = X.shape
    test_image = np.asarray([pictogram] * n_classes)
    test_image = test_image.reshape(n_classes, w, h, d)
    pairs = [test_image, X]
    #start_time = time.time()
    probs = model.predict(pairs)
    #print(f"Calculating probability in {(time.time() - start_time)} seconds.")
    #print("Probabilities: ")
    #print(probs)
    predicted = np.argmax(probs)
    #print(f"Id real: {folders[pictogram_num]}")
    print(f"Predicted id: {folders[predicted]}")
    print(probs[predicted])

    return folders[predicted], X[predicted]


def test_single_pictogram():

    predictions = dict()
    count = 0

    start_total_time = time.time()

    for photo in os.listdir("./test"):
        img = img_to_array(load_img(os.path.join("./test", photo), color_mode="rgba", target_size=TARGET_SIZE))
        print(f"Test photo: {os.path.join('./test', photo)}")
        while True:
            try:
                start_time = time.time()
                batchX_test, batchY_test = test_small_it.next()
                # show_batch_images(batchX_test)
                id, predicted_pictogram = test_pictogram_against_all(model, batchX_test, img)
                predictions[id] = predicted_pictogram
                # array_to_img(predictions[id]).show()
                print(f"Batch number: {count}")
                count += 1

                print(f"Time:{time.time() - start_time}")
                break
            except Exception:
                break

    print(f"Total time: {time.time() - start_total_time}")

def test_set_pictograms():
    global n_correct
    for j in range(10):
        print(f"Testing batch {j}")
        # batchX_test, batchY_test = test_small_it.next()
        batchX_test, batchY_test = test_real_it.next()
        # show_batch_images(batchX_test)
        for i in range(len(batchX_test)):
            n_correct += test_one_pictogram(model, batchX_test, i)

        break
    print(f"Correctly matched: {n_correct}")


#train_model()

#model.load_weights(os.path.join(model_path, last_weights_file)) #f'weights.{NUMBER_OF_ITERATIONS}.h5'))
#model.load_weights(os.path.join(model_path, f'weights.{50}.h5'))

#folders = {v: k for k, v in test_it.class_indices.items()}
#folders = {v: k for k, v in test_real_it.class_indices.items()}
#folders = {v: k for k, v in test_small_it.class_indices.items()}

#test_single_pictogram()
#batchX, batchy = train_it.next()
#show_batch_images(batchX)
model.summary()
