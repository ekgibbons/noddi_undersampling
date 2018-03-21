import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from keras.optimizers import Adam
from keras.utils.training_utils import multi_gpu_model
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint

from utils import readhd5
from utils import display

import models2d
import dense2d

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

n_gpu = 1
n_epochs = 200
batch_size = 10

image_size = (128,128,64)

# with tf.device("/cpu:0"):
model = dense2d.dense_net(image_size)
    
# model = multi_gpu_model(model, gpus=n_gpu)
    
model.compile(optimizer=Adam(lr=1e-3, beta_1=0.99, beta_2=0.995,
                             epsilon=1e-08, decay=0.85),
              loss="mean_absolute_error")

x_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/x_64_directions.h5"
y_odi_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_odi.h5"
y_fiso_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_fiso.h5"
y_ficvf_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_ficvf.h5"

print("Loading data...")
x = readhd5.ReadHDF5(x_path,"x_64_directions")
y_odi = readhd5.ReadHDF5(y_odi_path,"y_odi")
y_fiso = readhd5.ReadHDF5(y_fiso_path,"y_fiso")
y_ficvf = readhd5.ReadHDF5(y_ficvf_path,"y_ficvf")
print("Data is loaded...")

y = np.concatenate((y_odi, y_fiso, y_ficvf),
                   axis=2)

x /= np.amax(x)
y /= np.amax(y)

x = x.transpose(3,0,1,2)
y = y.transpose(3,0,1,2)

x_train = x[(x.shape[0]*4)//5:,:,:,:]
y_train = y[(y.shape[0]*4)//5:,:,:,:]

x_test = x[:(x.shape[0]*4)//5,:,:,:]
y_test = y[:(y.shape[0]*4)//5,:,:,:]

batch_size_multi_gpu = n_gpu*batch_size
train_steps = x_train.shape[0]//batch_size_multi_gpu
validation_steps = x_test.shape[0]//batch_size_multi_gpu


tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=1,
                          batch_size=batch_size_multi_gpu,
                          write_graph=True,
                          write_grads=True,
                          write_images=True,
                          embeddings_freq=0,
                          embeddings_layer_names=None,
                          embeddings_metadata=None)

checkpointer = ModelCheckpoint(filepath="/v/raid1b/egibbons/models/noddi-64.h5",
                               verbose=1,
                               save_best_only=True,
                               save_weights_only=True)

model.fit_generator(
    batch_generator(x_train, y_train,
                    batch_size_multi_gpu),
    steps_per_epoch=train_steps,
    epochs=n_epochs,
    verbose=1,
    validation_data=(x_test, y_test),
    validation_steps=validation_steps,
    callbacks=[checkpointer]
    )

x_test = x[0,:,:,:]
y_test = y[0,:,:,:]

recon = model.predict(x_test, batch_size=1)

print(recon.shape)

