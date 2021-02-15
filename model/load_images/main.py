from siamese_network import get_siamese_model
from one_shot import get_batch, test_oneshot
from keras.optimizers import Adam
import os

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model = get_siamese_model((105, 105, 4))
    optimizer = Adam(lr=0.00006)
    model.compile(loss="binary_crossentropy", optimizer=optimizer)
    evaluate_every = 1  # interval for evaluating on one-shot tasks
    batch_size = 1
    n_iter = 10 # No. of training iterations
    N_way = 1  # how many classes for testing one-shot tasks
    n_val = 10  # how many one-shot tasks to validate on
    best = -1
    model_path = './weights/'

    '''
    for i in range(1, n_iter + 1):
        print(i)
        (inputs, targets) = get_batch(batch_size)
        loss = model.train_on_batch(inputs, targets)
        if i % evaluate_every == 0:
            print("Train Loss: {0}".format(loss))
            val_acc = test_oneshot(model, N_way, n_val, verbose=True)
            model.save_weights(os.path.join(model_path, 'weights.{}.h5'.format(i)))
            if val_acc >= best:
                print("Current best: {0}, previous best: {1}".format(val_acc, best))
                best = val_acc
    '''

    model.load_weights(os.path.join(model_path, 'weights.10.h5'))
    val_acc = test_oneshot(model, N_way, n_val, verbose=True)
    print(val_acc)


    model.summary()
