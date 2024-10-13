import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from numpy.random import seed 
seed(1)
import tensorflow as tf
tf.random.set_seed(2)
import keras
from keras import models, layers

dataset = keras.preprocessing.image_dataset_from_directory(
    'training/',
    shuffle=True,
    image_size=(28, 28),
    batch_size=32,
    label_mode='categorical',
    color_mode='grayscale'
)

def get_dataset_partitions_tf(ds, train_split=0.9, test_split=0.1):
    ds_size = len(ds)

    train_size = int(train_split * ds_size)
    val_size = int(test_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    return train_ds, val_ds

train_ds, test_ds = get_dataset_partitions_tf(dataset)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

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
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

if __name__ == '__main__':
    model.fit(
        train_ds,
        validation_data=test_ds,
        verbose=2,
        epochs=12,
    )

model.export('mnist2')