import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import keras
from keras.optimizers import Adam
import numpy as np

sys.path.append("/home/mirl/egibbons/noddi")
from golkov_multi import model1d
from model_2d import dense2d
from noddi_utils import noddistudy
from noddi_utils import subsampling
from utils import display
from utils import readhd5

def model_2d(data,n_directions):

    image_size = (128,128,n_directions)
    loss_type = "l1"
    model = dense2d.dense_net(image_size)
    model.compile(optimizer=Adam(lr=1e-3),
                  loss="mean_absolute_error",
                  metrics=["accuracy"])
    model.load_weights("/v/raid1b/egibbons/models/noddi-%i_2d_%s.h5" %
                       (n_directions, loss_type))
    print("2D dense model loaded.  Using %s loss" % loss_type)

    
    max_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/max_values_2d.h5"
    maxs = readhd5.ReadHDF5(max_path,"max_values")[None,None,None,:]

    max_y_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/max_y_2d.h5"
    max_y = readhd5.ReadHDF5(max_y_path,"max_y")

    subsampling_pattern = subsampling.gensamples(n_directions)
    x = data[:,:,:,subsampling_pattern]
    x = x.transpose(2,0,1,3)

    x /= maxs.squeeze()[None,None,None,:]

    print("Predicting 2D...")
    start = time.time()
    prediction = model.predict(x, batch_size=1).transpose(1,2,0,3)
    print("Predictions completed...took: %f" % (time.time() - start))

    ### DISPLAY ###
    slice_use = 25
    
    for ii in range(4):
        prediction[:,:,:,ii] *= max_y[ii]

    return prediction
        
def golkov_multi(data, n_directions):

    max_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/max_values_1d.h5"
    maxs = readhd5.ReadHDF5(max_path,"max_values")[None,None,None,:]

    max_y_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/max_y_2d.h5"
    max_y = readhd5.ReadHDF5(max_y_path,"max_y")
    
    subsampling_pattern = subsampling.gensamples(n_directions)

    image_size = (n_directions,)

    model = model1d.fc_1d(image_size)
    model.compile(optimizer=Adam(lr=1e-3, beta_1=0.99, beta_2=0.995,
                                 epsilon=1e-08, decay=0.85),
                  loss="mean_absolute_error",
                  metrics=["accuracy"])
    model.load_weights("/v/raid1b/egibbons/models/noddi-%i_golkov_multi.h5"
                       % n_directions)
    print("golkov_multi model loaded")

    data_subsampled = data[:,:,:,subsampling_pattern]

    data_subsampled /= maxs

    dim0, dim1, n_slices, n_directions = data_subsampled.shape

    print("Predicting...")
    start = time.time()
    
    running_sum = data_subsampled.size
    for kk in range(n_slices):
        # running_sum += np.sum(data_subsampled[:,:,kk,0] > 1e-8)
        
        x = np.zeros((running_sum, n_directions))
        location = np.zeros((running_sum,3))
        
        ll = 0
        for ii in range(dim0):
            for jj in range(dim1):
                for kk in range(n_slices):
                    if data_subsampled[ii,jj,kk,0] > 1e-8:
                        x[ll,:] = data_subsampled[ii,jj,kk,:]
                        location[ll,:] = np.array([ii, jj, kk])
                        ll += 1

    recon = model.predict(x, batch_size=10000)

    prediction = np.zeros((dim0,dim1,n_slices,4))
    for ll in range(running_sum):
        ii = int(location[ll,0])
        jj = int(location[ll,1])
        kk = int(location[ll,2])
        prediction[ii,jj,kk,:] = recon[ll,:]

    for ii in range(4):
        prediction[:,:,:,ii] *= max_y[ii]
        
    print("Predictions completed...took: %f" % (time.time() - start))

    return prediction
