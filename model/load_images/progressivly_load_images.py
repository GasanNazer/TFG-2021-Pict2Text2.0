from keras_preprocessing.image import ImageDataGenerator, array_to_img

from siamese_network import get_siamese_model
#from one_shot import get_batch, test_oneshot, test
from keras.optimizers import Adam
import os
import numpy.random as rng
import numpy as np
from sklearn.utils import shuffle

datagen = ImageDataGenerator()

NUMBER_OF_ITERATIONS = 30
CLASSES = 3
NUM_IMAGES_PER_CLASS = 20 # used when constructing batches with pairs
BATCH_SIZE = CLASSES * NUM_IMAGES_PER_CLASS
TARGET_SIZE = (105,105)

# load and iterate training dataset
train_it = datagen.flow_from_directory('pictograms/0', target_size= TARGET_SIZE, color_mode="rgba", class_mode='categorical', batch_size=BATCH_SIZE, shuffle= True)
# load and iterate validation dataset
val_it = datagen.flow_from_directory('pictograms_val/0', target_size= TARGET_SIZE, color_mode="rgba", class_mode='categorical', batch_size=BATCH_SIZE, shuffle= False)
# load and iterate test dataset
test_it = datagen.flow_from_directory('pictograms_test', target_size= TARGET_SIZE, color_mode="rgba", class_mode='categorical', batch_size=BATCH_SIZE, shuffle= False)

##############################################
model = get_siamese_model(train_it.image_shape)
optimizer = Adam(lr=0.00006)
model.compile(loss="binary_crossentropy", optimizer=optimizer)
evaluate_every = 1  # interval for evaluating on one-shot tasks
#batch_size = 20
n_iter = 10 # No. of training iterations
N_way = 5  # how many classes for testing one-shot tasks
best = -1
model_path = './weights/'


def get_batch(batch_size, Xtrain, train_classes, s="train"):
    """
    Create batch of n pairs, half same class, half different class
    """

    # construct X in shape (num_classes, num_examples_per_class, w, h, d)

    num_classes = int(Xtrain.shape[0] / NUM_IMAGES_PER_CLASS)
    X = []
    for i in range(0, num_classes):
        X.append(Xtrain[20 * i:20 * (i + 1)])

    X = np.array(X)
    n_classes, n_examples, w, h, d = X.shape

    # randomly sample several classes to use in the batch
    if n_classes > batch_size:
        categories = rng.choice(n_classes, size=(batch_size,), replace=False)
    else:
        categories = rng.choice(n_classes, size=(n_classes,), replace=False)

    # initialize 2 empty arrays for the input image batch
    pairs = [np.zeros((batch_size, h, w, d)) for i in range(2)]

    # initialize vector for the targets
    targets = np.zeros((batch_size,))

    # make one half of it '1's, so 2nd half of batch has same class
    cat_1 = []
    cat_2 = []
    targets[batch_size // 2:] = 1
    for i in range(batch_size):
        cat_index = rng.randint(0, n_classes)
        category = categories[cat_index]
        idx_1 = rng.randint(0, n_examples)
        pairs[0][i, :, :, :] = X[category, idx_1].reshape(w, h, d)
        idx_2 = rng.randint(0, n_examples)

        # pick images of same class for 1st half, different for 2nd
        if i >= batch_size // 2:
            category_2 = category
        else:
            # add a random number to the category modulo n classes to ensure 2nd image has a different category
            category_2 = (category + rng.randint(1, n_classes)) % n_classes

        cat_1.append(category)
        cat_2.append(category_2)

        pairs[1][i, :, :, :] = X[category_2, idx_2].reshape(w, h, d)

    return pairs, targets


def show_batch_images(batch):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 8))
    columns = 6
    rows = 6
    count = 0
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        img = batch[count]
        plt.imshow(array_to_img(img))

        count += 1

        if len(batch) <= count:
            break

    plt.show()


def make_oneshot_task(batch_size, Xtrain, train_classes, N, s="val", pictogram=None):
    """Create pairs of test image, support set for testing N way one-shot learning. """

    num_classes = int(Xtrain.shape[0] / NUM_IMAGES_PER_CLASS)
    X = []
    for i in range(0, num_classes):
        X.append(Xtrain[20 * i:20 * (i + 1)])

    X = np.array(X)
    n_classes, n_examples, w, h, d = X.shape

    categories = train_classes

    indices = rng.randint(0, n_examples, size=(N,))
    if pictogram is not None:  # if pictogram is specified
        low, high = categories[pictogram]
        if N > high - low:
            raise ValueError("This pictogram ({}) has less than {} letters".format(pictogram, N))
        categories = rng.choice(range(low, high), size=(N,), replace=False)

    else:  # if no pictogram specified just pick a bunch of random ones
        if n_classes >= N:
            categories = rng.choice(range(N), size=(N,), replace=False)
        else:
            categories = rng.choice(range(n_classes), size=(n_classes,), replace=False)
            indices = rng.randint(0, n_examples, size=(n_classes,))
            N = n_classes

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

'''
for i in range(1, NUMBER_OF_ITERATIONS + 1):
    batchX, batchy = train_it.next()
    (inputs, targets) = get_batch(BATCH_SIZE, batchX, batchy)
    loss = model.train_on_batch(inputs, targets)

    batchX_val, batchY_val = val_it.next()
    inputs, targets = make_oneshot_task(BATCH_SIZE, batchX_val, batchY_val, N_way)
    probs = model.predict(inputs)
    #print(f"Probabilities: {probs}")


    if np.argmax(probs) == np.argmax(targets):
        n_correct += 1

    if i % 10 == 0:
        model.save_weights(os.path.join(model_path, 'weights.{}.h5'.format(i)))

    print(f"Iteration number: {i}")
'''

model.load_weights(os.path.join(model_path, 'weights.30.h5'))

batchX_test, batchY_test = train_it.next()

folders = {v: k for k, v in train_it.class_indices.items()}

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


print(f"Correctly matched: {n_correct}")

for i in range(7):
    test_one_pictogram(model, batchX_test, i)
model.summary()
