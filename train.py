import numpy as np

import tensorflow as tf
import keras
from keras.optimizers import Adam
from keras.utils.training_utils import multi_gpu_model
from keras import backend as K

from utils import readhd5

import models2d

# need to use this when I require batches from an already memory-loaded X, y
def batch_generator(X, y, batch_size):
    batch_i = 0
    while 1:
        if (batch_i+1)*batch_size >= len(X):
            yield X[batch_i*batch_size:], y[batch_i*batch_size:]
            batch_i = 0
        else:
            yield (X[batch_i*batch_size:(batch_i+1)*batch_size],
                   y[batch_i*batch_size:(batch_i+1)*batch_size])
            batch_i += 1

n_gpu = 2
n_epochs = 15
batch_size = 15

x_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/x_64_directions.h5"
y_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_odi.h5"

print("Loading data...")
x = readhd5.ReadHDF5(x_path,"x_64_directions")
y = readhd5.ReadHDF5(y_path,"y_odi")
print("Data is loaded...")

x /= np.amax(x)
y /= np.amax(y)

x = x.transpose(3,0,1,2)
y = y.transpose(3,0,1,2)

print(x.shape)
print(y.shape)

x_train = x[(x.shape[0]*4)//5:,:,:,:]
y_train = y[(y.shape[0]*4)//5:,:,:,:]

x_test = x[:(x.shape[0]*4)//5,:,:,:]
y_test = y[:(y.shape[0]*4)//5,:,:,:]

# print(x_train.shape)
# print(y_train.shape)

# print(x_test.shape)
# print(y_test.shape)

image_size = (128,128,64)

with tf.device("/cpu:0"):
    model = models2d.unet2d_model(image_size)
    
model = multi_gpu_model(model, gpus=n_gpu)
    
model.compile(optimizer=Adam(lr=1e-3, beta_1=0.99, beta_2=0.995,
                             epsilon=1e-08, decay=0.7),
              loss="mean_absolute_error",
              metrics=["accuracy"])

batch_size_multi_gpu = n_gpu*batch_size
train_steps = x_train.shape[0]//batch_size_multi_gpu
validation_steps = x_test.shape[0]//batch_size_multi_gpu


keras.callbacks.TensorBoard(log_dir='./logs',
                            histogram_freq=0,
                            batch_size=batch_size_multi_gpu,
                            write_graph=True,
                            write_grads=False,
                            write_images=False,
                            embeddings_freq=0,
                            embeddings_layer_names=None,
                            embeddings_metadata=None)


model.fit_generator(
    batch_generator(x_train, y_train,
                    batch_size_multi_gpu),
    train_steps,
    epochs=n_epochs,
    verbose=1,
    validation_data=batch_generator(x_test, y_test,
                                    batch_size_multi_gpu),
    validation_steps=validation_steps
    )

model.save_weights("/v/raid1b/egibbons/models/noddi-64.h5")

x_test = x[0,:,:,:]
y_test = y[0,:,:,:]

recon = model.predict(x_test, batch_size=1)

print(recon.shape)

