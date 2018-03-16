import numpy as np

import tensorflow as tf
import keras
from keras.optimizers import Adam
from keras.utils.training_utils import multi_gpu_model
from keras import backend as K

from utils import readhd5

import models2d

image_size = (128,128,64)
model = models2d.unet2d_model(image_size)
model.compile(optimizer=Adam(lr=1e-3, beta_1=0.99, beta_2=0.995,
                             epsilon=1e-08, decay=0.7),
              loss="mean_absolute_error",
              metrics=["accuracy"])
model.load_weights("/v/raid1b/egibbons/models/noddi-64.h5")

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

x_test = x[0,:,:,:]
y_test = y[0,:,:,:]

recon = model.predict(x_test, batch_size=1)

print(recon.shape)
