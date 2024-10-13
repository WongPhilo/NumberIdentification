import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import tensorflow as tf
import cv2

numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

model = tf.saved_model.load('mnist1')

def predict_number(image_name):
    img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = (img[...,::-1].astype(np.float32)) / 255.0
    img = img.reshape(1, 28, 28, 1)
    prediction = model.serve(img)
    return np.argmax(prediction)