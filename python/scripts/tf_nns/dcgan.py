from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    MaxPooling2D,
    Reshape,
    UpSampling2D,
)
from keras.models import Sequential


def generator_model():
    model = Sequential()
    model.add(Dense(1024))
    model.add(Activation("tanh"))
    model.add(Dense(128 * 7 * 7))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding="same"))
    model.add(Activation("tanh"))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding="same"))
    model.add(Activation("tanh"))
    return model


def discriminator_model():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), padding="same", input_shape=(28, 28, 1)))
    model.add(Activation("tanh"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation("tanh"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("tanh"))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    return model


def dcgan():
    d = discriminator_model()
    g = generator_model()
    model = Sequential()
    model.add(g)
    model.add(d)
    return model
