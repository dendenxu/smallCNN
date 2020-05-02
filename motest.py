from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2

model_path = "results/mo.h5"

model = load_model(model_path)
height, width = 192, 256


def predict(img):
    img = image.img_to_array(img)
    img = cv2.resize(img, (width, height))
    img = img * 1. / 255
    img = np.expand_dims(img, axis=0)
    labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    y_predict = model.predict(img)
    y_predict = labels[np.argmax(y_predict)]
    return y_predict