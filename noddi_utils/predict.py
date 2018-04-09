import os
import sys
import time

import keras
from keras.optimizers import Adam
import numpy as np

sys.path.append("/home/mirl/egibbons/noddi")
from golkov import model1d
from golkov_multi import model1d as model1d_multi
from model_2d import dense2d
from model_2d import simple2d
from model_2d import unet2d
from noddi_utils import noddistudy
from noddi_utils import subsampling
from utils import display
from utils import readhd5

def model_2d(data,n_directions,loss_type="l1"):

    image_size = (128,128,n_directions)

    # model = dense2d.dense_net(image_size)
    # model = simple2d.simple2d(image_size)
    model = simple2d.res2d(image_size)
    # model = unet2d.unet2d(image_size)
    model.compile(optimizer=Adam(lr=1e-3),
                  loss="mean_absolute_error",
                  metrics=["accuracy"])
    model.load_weights("/v/raid1b/egibbons/models/noddi-%i_2d_%s.h5" %
                       (n_directions, loss_type))
    print("2D dense model loaded.  Using %s loss" % loss_type)

    
    max_path = ("/v/raid1b/egibbons/MRIdata/DTI/noddi/max_values_%i_directions_2d.h5" %
                n_directions)
    maxs = readhd5.ReadHDF5(max_path,"max_values")[None,None,None,:]

    max_y_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/max_y_2d.h5"
    max_y = readhd5.ReadHDF5(max_y_path,"max_y")

    subsampling_pattern = subsampling.gensamples(n_directions)
    x = data[:,:,:,subsampling_pattern]
    x = x.transpose(2,0,1,3)

    x /= maxs.squeeze()[None,None,None,:]
    
    print("Predicting 2D...")
    start = time.time()
    prediction = model.predict(x, batch_size=10).transpose(1,2,0,3)
    print("Predictions completed...took: %f" % (time.time() - start))

    ### DISPLAY ###

    diffusivity_scaling = 1
    for ii in range(4):
        prediction[:,:,:,ii] *= max_y[ii]

    prediction[:,:,:,3] /= diffusivity_scaling
        
    return prediction
        
def golkov_multi(data, n_directions):

    max_path = ("/v/raid1b/egibbons/MRIdata/DTI/noddi/max_values_%i_directions_1d.h5" %
                n_directions)
    maxs = readhd5.ReadHDF5(max_path,"max_values")[None,None,None,:]

    max_y_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/max_y_1d.h5"
    max_y = readhd5.ReadHDF5(max_y_path,"max_y")
    
    subsampling_pattern = subsampling.gensamples(n_directions)

    image_size = (n_directions,)

    model = model1d_multi.fc_1d(image_size)
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
    
    # data_subsampled_temp = data_subsampled.reshape(0,1)
    x = data_subsampled.reshape(n_slices*dim0*dim1, -1)

    recon = model.predict(x, batch_size=10000)
    
    prediction = recon.reshape(dim0, dim1, n_slices, 4)
    
    for ii in range(4):
        prediction[:,:,:,ii] *= max_y[ii]
        
    print("Predictions completed...took: %f" % (time.time() - start))

    return prediction

def golkov(data, n_directions):

    max_path = ("/v/raid1b/egibbons/MRIdata/DTI/noddi/max_values_%i_directions_1d.h5" %
                n_directions)
    maxs = readhd5.ReadHDF5(max_path,"max_values")[None,None,None,:]

    max_y_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/max_y_1d.h5"
    max_y = readhd5.ReadHDF5(max_y_path,"max_y")
    
    subsampling_pattern = subsampling.gensamples(n_directions)

    image_size = (n_directions,)

    model = model1d.fc_1d(image_size)
    model.compile(optimizer=Adam(lr=1e-3, beta_1=0.99, beta_2=0.995,
                                 epsilon=1e-08, decay=0.85),
                  loss="mean_absolute_error",
                  metrics=["accuracy"])
    model.load_weights("/v/raid1b/egibbons/models/noddi-%i_golkov.h5"
                       % n_directions)
    print("golkov_multi model loaded")

    data_subsampled = data[:,:,:,subsampling_pattern]

    data_subsampled /= maxs

    dim0, dim1, n_slices, n_directions = data_subsampled.shape

    print("Predicting...")
    start = time.time()
    
    # data_subsampled_temp = data_subsampled.reshape(0,1)
    x = data_subsampled.reshape(n_slices*dim0*dim1, -1)

    recon = model.predict(x, batch_size=10000)
    
    prediction = recon.reshape(dim0, dim1, n_slices, 1)
    
    for ii in range(1):
        prediction[:,:,:,ii] *= max_y[ii]
        
    print("Predictions completed...took: %f" % (time.time() - start))

    return prediction
