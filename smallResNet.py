# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: 'Python 3.7.6 64-bit (''Anaconda'': conda)'
#     language: python
#     name: python37664bitanacondacondad8b493924e3244a48e8e41c0dd4f7d96
# ---

from keras.models import Model
from keras.layers import Dense, Input, LeakyReLU
from keras.callbacks import TensorBoard
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
import keras
import numpy as np
import matplotlib.pyplot as plt
log_dir = "./logs"  # Saving TensorBoard logs to logs dir 
data_n = int(1e2) # number of samples
# batch_size = 32
for i in range(100):
    if 2**i > data_n:
        batch_size = 2**(i-1)
        break
print(batch_size)
learning_rate = 1e-2
epoch_n = 64
depth = 8
scale = 32
layer_size = 32 # out = Dense(layer_size)(in)
x_train = np.random.rand(data_n, 1, 1) * scale # Random data in (0, 1)
y_train = x_train
x_val = np.random.rand(data_n//4, 1, 1) * scale
y_val = x_val

# Checking shapes for validation
print(x_val.shape)
print(x_train.shape)


# +
def DNN_IO(input_shape):
    inputs = Input(input_shape)
    dnn = Dense(layer_size,
               )(inputs)
    for _ in range(depth):
        dnn = Dense(layer_size,
                    activation="relu",
                   )(dnn)
#         dnn = LeakyReLU()(dnn)
        dnn = BatchNormalization()(dnn)
    outputs = Dense(1)(dnn)
    return inputs, outputs

def ResNet_IO(input_shape):
    inputs, outputs = DNN_IO(input_shape)
    outputs_res = keras.layers.add([inputs, outputs])
    return inputs, outputs_res

def DNN(input_shape):
    return CompileModel(*DNN_IO(input_shape))

def ResNet(input_shape):
    return CompileModel(*ResNet_IO(input_shape))

def CompileModel(inputs, outputs):
    model = Model(inputs, outputs)
    model.compile(loss='mean_squared_error',
#                   optimizer=Adam(lr=learning_rate),
                  optimizer=SGD(lr=learning_rate),
                  metrics=['mae'])
    return model

def FitModel(model):
    return model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epoch_n,
                    validation_data=(x_val, y_val),
                    callbacks=callbacks)

input_shape = (1, 1)
model = DNN(input_shape)
model_res = ResNet(input_shape)
tensorboard = TensorBoard(log_dir)
callbacks = [tensorboard, ]
history = FitModel(model)
history_res = FitModel(model_res)
# -

x_test = np.random.rand(data_n, 1, 1) * scale
y_test = x_test
y_pred = model.predict(x_test)
# print(y_pred)
y_pred_res = model_res.predict(x_test)
# print(y_pred_res)
x_test = x_test[:,0,0]
plt.figure(figsize=(10,10))
plt.scatter(x_test, y_test, label="Ground Truth")
plt.scatter(x_test, y_pred, label="DNN Prediction")
plt.scatter(x_test, y_pred_res, label="ResNet Prediction")
plt.legend()
plt.grid()
plt.show()
plt.savefig("smallResNet.eps")


