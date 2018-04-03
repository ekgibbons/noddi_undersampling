import os
import sys
import time 

loss_type = "l1"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from keras import backend as K
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.utils.training_utils import multi_gpu_model

sys.path.append("/home/mirl/egibbons/noddi")

import dense2d
import models2d
from noddi_utils import network_utils
import simple2d
import unet2d
from utils import readhd5
from utils import display


print("running network with %s loss" % loss_type)

if (len(sys.argv) == 1) or (sys.argv[1] == "64"):
    n_directions = 64
else:
    n_directions = int(sys.argv[1])

n_gpu = 1
n_epochs = 100
batch_size = 10
learning_rate = 1e-3

image_size = (128,128,n_directions)

# model = dense2d.dense_net(image_size)
# model = simple2d.simple2d(image_size)
model = simple2d.res2d(image_size)
# model = unet2d.unet2d(image_size)

optimizer = Adam(lr=learning_rate)
if loss_type == "l1":
    model.compile(optimizer=optimizer,
                  loss="mean_absolute_error")
else:
    model.compile(optimizer=optimizer,
                  loss=network_utils.perceptual_loss)


    
x_path = ("/v/raid1b/egibbons/MRIdata/DTI/noddi/x_%i_directions_2d.h5" %
          n_directions)
y_odi_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_odi_2d.h5"
y_fiso_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_fiso_2d.h5"
y_ficvf_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_ficvf_2d.h5"
y_gfa_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_gfa_2d.h5"
# y_md_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_md_2d.h5"
# y_ad_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_ad_2d.h5"
# y_fa_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_fa_2d.h5"

print("Loading data...")

start = time.time()
y_odi = readhd5.ReadHDF5(y_odi_path,"y_odi")
y_fiso = readhd5.ReadHDF5(y_fiso_path,"y_fiso")
y_ficvf = readhd5.ReadHDF5(y_ficvf_path,"y_ficvf")
y_gfa = readhd5.ReadHDF5(y_gfa_path,"y_gfa")
# y_md = readhd5.ReadHDF5(y_md_path,"y_md")
# y_ad = readhd5.ReadHDF5(y_ad_path,"y_ad")
# y_fa = readhd5.ReadHDF5(y_fa_path,"y_fa")

diffusivity_scaling = 1
y = np.concatenate((y_odi, y_fiso, y_ficvf,
                    diffusivity_scaling*y_gfa),
                    # diffusivity_scaling*y_md,
                    # diffusivity_scaling*y_ad,
                    # y_fa),
                   axis=3)

x = readhd5.ReadHDF5(x_path,"x_%i_directions" % n_directions)

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

save_path = ("/v/raid1b/egibbons/models/noddi-%i_2d_%s.h5"
             % (n_directions, loss_type))
print("saving to: %s" % save_path)
checkpointer = ModelCheckpoint(filepath=save_path,
                               verbose=1,
                               save_best_only=True,
                               save_weights_only=True,
                               period=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=5, min_lr=1e-7) 

lrate = LearningRateScheduler(network_utils.step_decay)

model.fit(x=x,
          y=y,
          batch_size=batch_size_multi_gpu,
          epochs=n_epochs,
          verbose=1,
          callbacks=[checkpointer, lrate],
          validation_split=0.2,
          shuffle=True,
          )
