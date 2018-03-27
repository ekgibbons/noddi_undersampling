import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from keras.optimizers import Adam, SGD
from keras.utils.training_utils import multi_gpu_model
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

import model1d
from utils import display
from utils import readhd5



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
n_epochs = 1000
batch_size = 10000
learning_rate = 1e-3

image_size = (64,)

model = model1d.fc_1d(image_size)
    
optimizer = Adam(lr=learning_rate)

model.compile(optimizer=optimizer,
              loss="mean_absolute_error",
              )

x_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/x_64_directions_train_1d.h5"
y_odi_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_odi_train_1d.h5"
y_fiso_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_fiso_train_1d.h5"
y_ficvf_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_ficvf_train_1d.h5"

print("Loading data...")
x = readhd5.ReadHDF5(x_path,"x_64_directions")
y_odi = readhd5.ReadHDF5(y_odi_path,"y_odi")
y_fiso = readhd5.ReadHDF5(y_fiso_path,"y_fiso")
y_ficvf = readhd5.ReadHDF5(y_ficvf_path,"y_ficvf")
print("Data is loaded...")

n_samples, _ = x.shape

y = np.concatenate((y_odi, y_fiso, y_ficvf),
                   axis=1)

print(y_odi.shape)
print(y.shape)
print(x.shape)

plt.figure()
plt.imshow(y[:64,:])
plt.show()

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

checkpointer = ModelCheckpoint(filepath="/v/raid1b/egibbons/models/noddi-64_1d.h5",
                               verbose=1, monitor="val_loss", save_best_only=True,
                               save_weights_only=True, period=25)

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

