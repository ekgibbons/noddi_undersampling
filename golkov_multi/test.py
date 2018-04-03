import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

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
from noddi_utils import predict
from noddi_utils import subsampling

n_channels = 64

max_y_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/max_y_1d.h5"
max_y = readhd5.ReadHDF5(max_y_path,"max_y")


patient_number = "P111816"
noddi_data = noddistudy.NoddiData(patient_number)

data_full = noddi_data.get_full()
data_odi = noddi_data.get_odi()
data_fiso = noddi_data.get_fiso()
data_ficvf = noddi_data.get_ficvf()
data_gfa = noddi_data.get_gfa()
# data_md = noddi_data.get_md()
# data_ad = noddi_data.get_ad()
# data_fa = noddi_data.get_fa()

data_odi[data_odi>max_y[0]] = max_y[0]
data_fiso[data_fiso>max_y[1]] = max_y[1]
data_ficvf[data_ficvf>max_y[2]] = max_y[2]
data_gfa[data_gfa>max_y[3]] = max_y[3]
# data_md[data_md>max_y[4]] = max_y[4]
# data_ad[data_ad>max_y[5]] = max_y[5]
# data_fa[data_fa>max_y[6]] = max_y[6]

prediction = predict.golkov_multi(data_full, n_channels)

slice_use = 25

montage_top = np.concatenate((data_odi[:,:,slice_use],
                              data_fiso[:,:,slice_use],
                              data_ficvf[:,:,slice_use]),
                             axis=1)

montage_bottom = np.concatenate((prediction[:,:,slice_use,0],
                                 prediction[:,:,slice_use,1],
                                 prediction[:,:,slice_use,2]),
                                axis=1)

montage_bottom *= (montage_top > 0)

montage_combine = np.concatenate((montage_top,
                                  montage_bottom,
                                  abs(montage_top - montage_bottom)),
                                 axis=0)

plt.figure()
plt.imshow(montage_combine)
plt.title("ODI, FISO, FICVF")
plt.axis("off")

plt.figure()
plt.imshow(np.concatenate((data_gfa[:,:,slice_use],
                           prediction[:,:,slice_use,3],
                           abs(data_gfa[:,:,slice_use] - prediction[:,:,slice_use,3])),
                          axis=0), vmax=0.5)
plt.title("GFA")
plt.axis("off")

# plt.figure()
# plt.imshow(np.concatenate((data_md[:,:,slice_use],
#                            prediction[:,:,slice_use,4],
#                            abs(data_md[:,:,slice_use] - prediction[:,:,slice_use,4])),
#                           axis=0), vmax=3e-3)
# plt.title("MD")
# plt.axis("off")

# plt.figure()
# plt.imshow(np.concatenate((data_ad[:,:,slice_use],
#                            prediction[:,:,slice_use,5],
#                            abs(data_ad[:,:,slice_use] - prediction[:,:,slice_use,5])),
#                           axis=0), vmax=3e-3)
# plt.title("AD")
# plt.axis("off")

# plt.figure()
# plt.imshow(np.concatenate((data_fa[:,:,slice_use],
#                            prediction[:,:,slice_use,6],
#                            abs(data_fa[:,:,slice_use] - prediction[:,:,slice_use,6])),
#                           axis=0), vmax=0.7)
# plt.title("FA")
# plt.axis("off")

plt.show()













# subsampling = subsampling.gensamples(n_channels)

# image_size = (len(subsampling),)

# model = model1d.fc_1d(image_size)
# model.compile(optimizer=Adam(lr=1e-3, beta_1=0.99, beta_2=0.995,
#                              epsilon=1e-08, decay=0.85),
#               loss="mean_absolute_error",
#               metrics=["accuracy"])
# model.load_weights("/v/raid1b/egibbons/models/noddi-%i_golkov_multi.h5"
#                    % n_channels)
# print("Model loaded")

# data_subsampled = data_full[:,:,:,subsampling]

# data_subsampled /= maxs

# dim0, dim1, n_slices, n_channels = data_subsampled.shape

# print("Predicting...")
# start = time.time()

# running_sum = 0
# for kk in range(n_slices):
#     running_sum += np.sum(data_subsampled[:,:,kk,0] > 1e-8)

# x = np.zeros((running_sum, n_channels))
# location = np.zeros((running_sum,3))

# ll = 0
# for ii in range(dim0):
#     for jj in range(dim1):
#         for kk in range(n_slices):
#             if data_subsampled[ii,jj,kk,0] > 1e-8:
#                 x[ll,:] = data_subsampled[ii,jj,kk,:]
#                 location[ll,:] = np.array([ii, jj, kk])
#                 ll += 1

# recon = model.predict(x, batch_size=10000)

# prediction = np.zeros((dim0,dim1,n_slices,4))
# for ll in range(running_sum):
#     ii = int(location[ll,0])
#     jj = int(location[ll,1])
#     kk = int(location[ll,2])
#     prediction[ii,jj,kk,:] = recon[ll,:]
    
# print("Predictions completed...took: %f" % (time.time() - start))

# for ii in range(4):
#     prediction[:,:,:,ii] *= max_y[ii]
