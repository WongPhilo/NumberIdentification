import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from numpy.random import seed 
seed(1)
import keras
from keras import models, layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Resizing(28, 28),
    layers.Rescaling(1./255),
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax'),
])

model.build(input_shape=(None, 28, 28, 1))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])

if __name__ == '__main__':
    model.fit(
        x_train,
        y_train,
        batch_size=32,
        epochs=16,
        verbose=2,
        validation_split=0.1
    )

model.export('mnist1')