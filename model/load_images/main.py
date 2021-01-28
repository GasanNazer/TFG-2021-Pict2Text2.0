from siamese_network import get_siamese_model
from keras.optimizers import Adam

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model = get_siamese_model((105, 105, 1))
    optimizer = Adam(lr=0.00006)
    model.compile(loss="binary_crossentropy", optimizer=optimizer)
    model.summary()
