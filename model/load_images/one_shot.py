from keras_preprocessing.image import array_to_img
from sklearn.utils import shuffle
import numpy.random as rng
import numpy as np

from load_images import load_images, show_image_from_array, load_test_images

X, Y, folders = load_images()
Xtrain = X
train_classes = Y
Xval, Y_val, folders_val = load_images("pictograms_val", classes_loaded= len(folders))
val_classes = Y_val

def get_batch(batch_size, s="train"):
    """
    Create batch of n pairs, half same class, half different class
    """
    if s == 'train':
        X = Xtrain
        categories = train_classes
    else:
        X = Xval
        categories = val_classes
    n_classes, n_examples, w, h, d = X.shape

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
        idx_1 = rng.randint(0, n_examples)
        pairs[0][i, :, :, :] = X[category, idx_1].reshape(w, h, d)
        idx_2 = rng.randint(0, n_examples)

        # pick images of same class for 1st half, different for 2nd
        if i >= batch_size // 2:
            category_2 = category
        else:
            # add a random number to the category modulo n classes to ensure 2nd image has a different category
            category_2 = (category + rng.randint(1, n_classes)) % n_classes

        pairs[1][i, :, :, :] = X[category_2, idx_2].reshape(w, h, d)

    return pairs, targets

def generate(batch_size, s="train"):
    """
    a generator for batches, so model.fit_generator can be used.
    """
    while True:
        pairs, targets = get_batch(batch_size,s)
        yield (pairs, targets)

def make_oneshot_task(N, s="val", pictogram=None):
    """Create pairs of test image, support set for testing N way one-shot learning. """
    if s == 'train':
        X = Xtrain
        categories = train_classes
    else:
        X = Xval
        categories = val_classes
    n_classes, n_examples, w, h, d = X.shape

    indices = rng.randint(0, n_examples, size=(N,))
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


def test_oneshot(model, N, k, s = "val", verbose = 0):
    """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
    n_correct = 0
    if verbose:
        print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k,N))
    for i in range(k):
        inputs, targets = make_oneshot_task(N,s)
        probs = model.predict(inputs)
        if np.argmax(probs) == np.argmax(targets):
            n_correct+=1
    percent_correct = (100.0 * n_correct / k)
    if verbose:
        print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct,N))
    return percent_correct


def test(model):
    input = Xval
    #show_image_from_array(Xval, 0)
    val = Y_val[0][0]
    a = np.argmax(val)
    print(f"Id: {folders_val[np.argmax(val)]}")

    val = Y_val[25][0]
    a = np.argmax(val)
    print(f"Id: {folders_val[np.argmax(val)]}")
    show_image_from_array(Xval, 1)
    input = np.expand_dims(Xval[0][0], axis= 0) #take all images class 0 / take the 1 image of the class 0
    input2 = np.expand_dims(Xval[1][0], axis= 0) #take all images class 1 / take the 1 image of the clase 1
    inputs = []
    inputs.append(input)
    inputs.append(input2)
    a = model.predict(inputs)
    print(a)

