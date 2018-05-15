import os
import sys
import time 

loss_type = "l1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import keras
from keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot as plt

sys.path.append("/home/mirl/egibbons/noddi")

from model_2d import simple2d
from noddi_utils import noddistudy
from noddi_utils import subsampling
from recon import imtools
from utils import readhd5

test_cases = ["P032315","P080715","P061114",
              "N011118A","N011118B"]

n_directions = 24

sampling_pattern = subsampling.gensamples(n_directions)

### LOAD DATA ###
patient_number = test_cases[0]
noddi_data = noddistudy.NoddiData(patient_number)

data_full = noddi_data.get_full()
data_odi = noddi_data.get_odi().transpose(1,0,2)[::-1,::-1]
data_fiso = noddi_data.get_fiso().transpose(1,0,2)[::-1,::-1]
data_ficvf = noddi_data.get_ficvf().transpose(1,0,2)[::-1,::-1]
data_gfa = noddi_data.get_gfa().transpose(1,0,2)[::-1,::-1]

max_y_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/max_y_2d.h5"
max_y = readhd5.ReadHDF5(max_y_path,"max_y")

### LOAD MODEL ###

image_size = (128,128,n_directions)

model = simple2d.res2d(image_size)
model.compile(optimizer=Adam(lr=1e-3),
              loss="mean_absolute_error",
              metrics=["accuracy"])
model.load_weights("noddi_test-%i_2d.h5" %
                   (n_directions))
print("2D dense model loaded.  Using %s loss" % loss_type)

    
max_path = ("/v/raid1b/egibbons/MRIdata/DTI/noddi/"
            "max_values_%i_directions_2d.h5" %
            n_directions)
maxs = readhd5.ReadHDF5(max_path,"max_values")[None,None,None,:]

max_y_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/max_y_2d.h5"
max_y = readhd5.ReadHDF5(max_y_path,"max_y")

subsampling_pattern = subsampling.gensamples(n_directions)
x = data_full[:,:,:,subsampling_pattern]
x = x.transpose(2,0,1,3)

x /= maxs.squeeze()[None,None,None,:]

print("Predicting 2D...")
start = time.time()
prediction_2d = model.predict(x, batch_size=10).transpose(2,1,0,3)[::-1,::-1]
print("Predictions completed...took: %f" % (time.time() - start))

### DISPLAY ###
for ii in range(4):
    prediction_2d[:,:,:,ii] *= max_y[ii]

slice_use = 22

montage_1 = np.concatenate((data_odi[:,:,slice_use],
                            data_fiso[:,:,slice_use],
                            data_ficvf[:,:,slice_use],
                            data_gfa[:,:,slice_use]/0.5),
                           axis=1)

montage_2 = np.concatenate((prediction_2d[:,:,slice_use,0],
                            prediction_2d[:,:,slice_use,1],
                            prediction_2d[:,:,slice_use,2],
                            prediction_2d[:,:,slice_use,3]/0.5),
                           axis=1)
    
montage_combine = np.concatenate((montage_1,
                                  montage_2,
                                  abs(montage_1 - montage_2)),
                                 axis=0)

plt.figure()
plt.imshow(montage_combine,cmap="gray")
plt.title("ODI, FISO, FICVF, GFA")
plt.axis("off")

plt.show()
    
    
