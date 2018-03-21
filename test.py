import time

import keras
from keras.optimizers import Adam
from keras import backend as K
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from utils import readhd5

import models2d
import noddistudy

subsampling = [0, 1, 2, 3, 5, 6, 8, 10, 11, 12, 13, 15, 18, 21, 27, 31,
               32, 40, 41, 45, 47, 49, 52, 55, 57, 60, 63, 65, 66, 70,
               81, 82, 86, 94, 95, 99, 100, 102, 104, 107, 110, 113, 115,
               118, 123, 130, 135, 140, 145, 150, 155, 160, 164, 167,
               168, 174, 180, 184, 187, 190, 193, 197, 200, 205]

image_size = (128,128,64)
model = models2d.unet2d_model(image_size)
model.compile(optimizer=Adam(lr=1e-3, beta_1=0.99, beta_2=0.995,
                             epsilon=1e-08, decay=0.7),
              loss="mean_absolute_error",
              metrics=["accuracy"])
model.load_weights("/v/raid1b/egibbons/models/noddi-64.h5")
print("Model loaded")

patient_number = "N011618"
noddi_data = noddistudy.NoddiData(patient_number)

data_full = noddi_data.get_full()
data_odi = noddi_data.get_odi()
data_fiso = noddi_data.get_fiso()
data_ficvf = noddi_data.get_ficvf()

x = data_full[:,:,:,subsampling]

x_max = np.amax(x)
x /= x_max

x = x.transpose(2,0,1,3)
print(x.shape)

print("Predicting...")
start = time.time()
recon = model.predict(x, batch_size=32)
print("Predictions completed...took: %f" % (time.time() - start))

slice_use = 25

montage_top = np.concatenate((data_odi[:,:,slice_use],
                              data_fiso[:,:,slice_use],
                              data_ficvf[:,:,slice_use]),
                             axis=1)

montage_bottom = np.concatenate((recon[slice_use,:,:,0],
                                 recon[slice_use,:,:,1],
                                 recon[slice_use,:,:,2]),
                                axis=1)

montage_combine = np.concatenate((montage_top,
                                  montage_bottom),
                                 axis=0)

plt.figure()
plt.imshow(montage_combine)
plt.show()

