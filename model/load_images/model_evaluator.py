from load_images import load_test_images
import numpy as np
import time

import concurrent
from itertools import repeat

start_time = time.time()

Xtest, folders_test = load_test_images(MAX=1000)  # (#pictograms, 105,105,4) # should take next 200 and next...

print(f"Loading images in {(time.time() - start_time)} seconds.")

def test_one_pictogram(model, X = Xtest, pictogram = Xtest[5], pictogram_num = 5):
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
    print(f"Id real: {folders_test[pictogram_num]}")
    print(f"Id: {folders_test[predicted]}")
    print(probs[predicted])


def concurent_checker(model, pictogram_num):
    count = 100
    X = []

    pictogram = Xtest[pictogram_num]

    for i in range(10):
        X.append(Xtest[count * i: count * (i + 1)])

        if count * (i + 1) >= Xtest.shape[0]:
            break

    start_time = time.time()
    #test_one_pictogram(model, Xtest, pictogram, pictogram_num)


    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        thread = executor.map(test_one_pictogram, repeat(model), X, repeat(pictogram), repeat(pictogram_num))

    print(f"Calculating probability in {(time.time() - start_time)} seconds.")

