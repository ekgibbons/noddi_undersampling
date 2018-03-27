import os
import time 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from keras.optimizers import Adam
from keras.utils.training_utils import multi_gpu_model
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau

from utils import readhd5
from utils import display

import models2d
import dense2d

n_gpu = 1
n_epochs = 200
batch_size = 20
learning_rate = 1e-3

image_size = (128,128,64)

model = dense2d.dense_net(image_size)
    
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss="mean_absolute_error")

x_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/x_64_directions_2d.h5"
y_odi_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_odi_2d.h5"
y_fiso_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_fiso_2d.h5"
y_ficvf_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_ficvf_2d.h5"

print("Loading data...")

start = time.time()
y_odi = readhd5.ReadHDF5(y_odi_path,"y_odi")
y_fiso = readhd5.ReadHDF5(y_fiso_path,"y_fiso")
y_ficvf = readhd5.ReadHDF5(y_ficvf_path,"y_ficvf")

y = np.concatenate((y_odi, y_fiso, y_ficvf),
                   axis=3)

x = readhd5.ReadHDF5(x_path,"x_64_directions")

print("Data is loaded...took: %f seconds" % (time.time() - start))

batch_size_multi_gpu = n_gpu*batch_size

tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=1,
                          batch_size=batch_size_multi_gpu,
                          write_graph=True,
                          write_grads=True,
                          write_images=True,
                          embeddings_freq=0,
                          embeddings_layer_names=None,
                          embeddings_metadata=None)

checkpointer = ModelCheckpoint(filepath="/v/raid1b/egibbons/models/noddi-64_2d.h5",
                               verbose=1,
                               save_best_only=True,
                               save_weights_only=True,
                               period=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=5, min_lr=1e-7) 

model.fit(x=x,
          y=y,
          batch_size=batch_size_multi_gpu,
          epochs=n_epochs,
          verbose=1,
          callbacks=[checkpointer, reduce_lr],
          validation_split=0.2,
          shuffle=True,
          )
