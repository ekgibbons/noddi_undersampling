import os
import sys
import time

import keras
from keras.optimizers import Adam
from keras import backend as K
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from utils import display
from utils import readhd5


sys.path.append("/home/mirl/egibbons/noddi")

import model1d
from noddi_utils import noddistudy



subsampling = [0, 1, 2, 3, 5, 6, 8, 10, 11, 12, 13, 15, 18, 21, 27, 31,
               32, 40, 41, 45, 47, 49, 52, 55, 57, 60, 63, 65, 66, 70,
               81, 82, 86, 94, 95, 99, 100, 102, 104, 107, 110, 113, 115,
               118, 123, 130, 135, 140, 145, 150, 155, 160, 164, 167,
               168, 174, 180, 184, 187, 190, 193, 197, 200, 205]

image_size = (len(subsampling),)

max_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/max_values_1d.h5"
maxs = readhd5.ReadHDF5(max_path,"max_values")[None,None,None,:]


model = model1d.fc_1d(image_size)
model.compile(optimizer=Adam(lr=1e-3, beta_1=0.99, beta_2=0.995,
                             epsilon=1e-08, decay=0.85),
              loss="mean_absolute_error",
              metrics=["accuracy"])
model.load_weights("/v/raid1b/egibbons/models/noddi-64_golkov.h5")
print("Model loaded")

patient_number = "N011618"
noddi_data = noddistudy.NoddiData(patient_number)

data_full = noddi_data.get_full()
data_odi = noddi_data.get_odi()
data_fiso = noddi_data.get_fiso()
data_ficvf = noddi_data.get_ficvf()

data_subsampled = data_full[:,:,:,subsampling]

plt.figure()
display.Render(data_subsampled[:,:,25,:])


#data_subsampled = (data_subsampled - means)/stds
data_subsampled /= maxs

plt.figure()
display.Render(data_subsampled[:,:,25,:])

dim0, dim1, n_slices, n_channels = data_subsampled.shape


print("Predicting...")
start = time.time()

running_sum = 0
for kk in range(n_slices):
    running_sum += np.sum(data_subsampled[:,:,kk,0] > 1e-8)

x = np.zeros((running_sum, n_channels))
location = np.zeros((running_sum,3))

ll = 0
for ii in range(dim0):
    for jj in range(dim1):
        for kk in range(n_slices):
            if data_subsampled[ii,jj,kk,0] > 1e-8:
                x[ll,:] = data_subsampled[ii,jj,kk,:]
                location[ll,:] = np.array([ii, jj, kk])
                ll += 1

# x_max = np.amax(x)
# x /= x_max

recon = model.predict(x, batch_size=10000)

prediction = np.zeros((dim0,dim1,n_slices,1))
for ll in range(running_sum):
    ii = int(location[ll,0])
    jj = int(location[ll,1])
    kk = int(location[ll,2])
    prediction[ii,jj,kk,:] = recon[ll,0]

print("Predictions completed...took: %f" % (time.time() - start))

slice_use = 25

# montage_top = np.concatenate((data_odi[:,:,slice_use],
#                               data_fiso[:,:,slice_use],
#                               data_ficvf[:,:,slice_use]),
#                              axis=1)

# montage_bottom = np.concatenate((prediction[:,:,slice_use,0],
#                                  prediction[:,:,slice_use,1],
#                                  prediction[:,:,slice_use,2]),
#                                 axis=1)


montage_top = data_odi[:,:,slice_use]
montage_bottom = prediction[:,:,slice_use,0]


montage_combine = np.concatenate((montage_top,
                                  montage_bottom),
                                 axis=0)

plt.figure()
plt.imshow(montage_combine)

plt.figure()
plt.imshow(prediction[:,:,slice_use,0])

plt.show()

