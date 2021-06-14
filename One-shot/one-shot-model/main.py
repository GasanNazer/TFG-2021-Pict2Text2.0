from siamese_network import get_siamese_model
from one_shot import get_batch
from keras.optimizers import Adam
from load_images import load_images
from progressivly_load_images import test_pictogram_against_all
from keras_preprocessing.image import ImageDataGenerator, array_to_img, load_img, img_to_array
import os

TARGET_SIZE = (105, 105)
batchX_val, batchY_val, folders = load_images("pictograms_val_digital")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model = get_siamese_model((105, 105, 4))
    optimizer = Adam(lr=0.00006)
    model.compile(loss="binary_crossentropy", optimizer=optimizer)
    evaluate_every = 1 #50  # interval for evaluating on one-shot tasks
    batch_size = 2 #111
    n_iter = 1 #2000 # No. of training iterations
    N_way = 20  # how many classes for testing one-shot tasks
    n_val = 5  # how many one-shot tasks to validate on
    best = -1
    model_path = './weights'
    inputs, targets = get_batch(batch_size)
    for i in range(1, n_iter + 1):
        print("iteration: " + str(i))
        (inputs, targets) = get_batch(batch_size)
        m = model.fit(inputs, targets, batch_size=batch_size, verbose=0, validation_split=0.1)

        if i % evaluate_every == 0:
            val_acc = 0
            for filename in os.listdir("./pictograms_val"):
                img = img_to_array(load_img(os.path.join("./pictograms_val", filename), color_mode="rgba", target_size=TARGET_SIZE))
                id, predicted_pictogram = test_pictogram_against_all(model, batchX_val[:, 0, :, :, :], img, folders)
                print("iteration: " + str(i))
                print(id)
                print(filename)
                if str(id) in filename:
                    val_acc += 1
                    print(val_acc)
            model.save_weights(os.path.join(model_path, 'weights.{}.h5'.format(i)))
            print("accuracy: ")
            val_acc /= 20
            print(val_acc)
            print("loss: ")
            print(m.history['loss'])
            if val_acc >= best:
                print("Current best: {0}, previous best: {1}".format(val_acc, best))
                best = val_acc
model.summary()



'''
#print("Train Loss: {0}".format(loss))
            #val_acc = test_oneshot(model, N_way, n_val, verbose=True)
            #model.save_weights(os.path.join(model_path, 'weights.{}.h5'.format(i)))
'''
