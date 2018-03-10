import numpy as np

import tensorflow as tf
from keras.optimizers import Adam
from keras.utils.training_utils import multi_gpu_model
from keras import backend as K

from utils import readhd5

import unet3d

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
batch_size = 30

x_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/x_64_directions.h5"
y_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_odi.h5"

x = readhd5.ReadHDF5(x_path,"x_64_directions")
y = readhd5.ReadHDF5(y_path,"y_odi")
print("Data is loaded...")


x /= np.amax(x)
y /= np.amax(y)

print(K.image_data_format())
x = x.transpose(3,0,1,2)
y = y.transpose(3,0,1,2)

x_train = x[(x.shape[3]*4)//5:,:,:,:,None]
y_train = y[(y.shape[3]*4)//5:,:,:,:,None]

x_test = x[:(x.shape[3]*4)//5,:,:,:,None]
y_test = y[:(y.shape[3]*4)//5,:,:,:,None]
print(x_train.shape)

image_size = (128,128,64,1)

with tf.device("/cpu:0"):
    model = unet3d.unet3d_model(image_size)
    
model = multi_gpu_model(model, gpus=n_gpu)
    
model.compile(optimizer=Adam(lr=1e-3, beta_1=0.99, beta_2=0.995,
                             epsilon=1e-08, decay=0.0),
              loss="mean_absolute_error",
              metrics=["accuracy"])

model.fit_generator(
    batch_generator(x_train,y_train, batch_size),
    x_train.shape[0]//batch_size,
    epochs=n_epochs,
    verbose=1,
    validation_data=batch_generator(x_test, y_test, batch_size),
    validation_steps=x_test.shape[0]//batch_size
    )

model.save_weights("/v/raid1b/egibbons/models/noddi-64.h5")

# TODO:

# python train.py 
# Using TensorFlow backend.
# Data is loaded...
# channels_last
# (1692, 128, 128, 64, 1)
# Epoch 1/15
# Traceback (most recent call last):
#   File "train.py", line 68, in <module>
#     validation_steps=x_test.shape[0]//batch_size
#   File "/home/mirl/egibbons/.conda/envs/ekg/lib/python3.5/site-packages/keras/legacy/interfaces.py", line 91, in wrapper
#     return func(*args, **kwargs)
#   File "/home/mirl/egibbons/.conda/envs/ekg/lib/python3.5/site-packages/keras/engine/training.py", line 2244, in fit_generator
#     class_weight=class_weight)
#   File "/home/mirl/egibbons/.conda/envs/ekg/lib/python3.5/site-packages/keras/engine/training.py", line 1884, in train_on_batch
#     class_weight=class_weight)
#   File "/home/mirl/egibbons/.conda/envs/ekg/lib/python3.5/site-packages/keras/engine/training.py", line 1487, in _standardize_user_data
#     exception_prefix='target')
#   File "/home/mirl/egibbons/.conda/envs/ekg/lib/python3.5/site-packages/keras/engine/training.py", line 113, in _standardize_input_data
#     'with shape ' + str(data_shape))
# ValueError: Error when checking target: expected conv2d_3 to have 4 dimensions, but got array with shape (30, 128, 128, 1, 1)

# Compilation exited abnormally with code 1 at Sat Mar 10 01:04:43


