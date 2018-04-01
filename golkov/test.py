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
from noddi_utils import subsampling

subsampling = subsampling.gensamples(64)

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
    running_sum += np.sum(data_subsampled[:,:,kk,0] > 1e-4)

x = np.zeros((running_sum, n_channels))
location = np.zeros((running_sum,3))

ll = 0
for ii in range(dim0):
    for jj in range(dim1):
        for kk in range(n_slices):
            if data_subsampled[ii,jj,kk,0] > 1e-4:
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


prediction *= (data_odi[:,:,:,None] > 0)

montage_top = data_odi[:,:,slice_use]
montage_bottom = prediction[:,:,slice_use,0]


plt.figure()
plt.imshow(np.concatenate((montage_top,
                           montage_bottom,
                           abs(montage_top - montage_bottom)),
                          axis=0))


plt.show()

