import os
import sys
import time

import keras
from keras.optimizers import Adam
import numpy as np

sys.path.append("/home/mirl/egibbons/noddi")
from gfa_2d import simple2d as simple2d_gfa
from golkov_multi import model1d as model1d_multi
from model_1d import simple2d as simple1d
from model_2d import simple2d
from noddi_2d import simple2d as simple2d_noddi
from noddi_utils import noddistudy
from noddi_utils import subsampling
from utils import display
from utils import readhdf5

def model_2d(data,n_directions,random_seed=400,loss_type="l1"):

    image_size = (128,128,n_directions)

    model = simple2d.res2d(image_size)
    model.compile(optimizer=Adam(lr=1e-3),
                  loss="mean_absolute_error",
                  metrics=["accuracy"])
    model.load_weights("/v/raid1b/egibbons/models/noddi-%i_2d_%s.h5" %
                       (n_directions, loss_type))
    print("2D dense model loaded.  Using %s loss" % loss_type)

    
    max_path = ("/v/raid1b/egibbons/MRIdata/DTI/noddi/max_values_%i_directions_2d.h5" %
                n_directions)
    maxs = readhdf5.read_hdf5(max_path,"max_values")[None,None,None,:]

    max_y_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/max_y_2d.h5"
    max_y = readhdf5.read_hdf5(max_y_path,"max_y")

    subsampling_pattern = subsampling.gensamples(n_directions,random_seed=random_seed)
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

def model_raw(data,n_directions,random_seed=400,loss_type="l1"):

    image_size = (128,128,n_directions)

    model = simple2d.res2d(image_size)
    model.compile(optimizer=Adam(lr=1e-3),
                  loss="mean_absolute_error",
                  metrics=["accuracy"])
    model.load_weights("/v/raid1b/egibbons/models/noddi-%i_raw.h5" %
                       (n_directions))
    print("2D dense model loaded for raw data.  Using %s loss" % loss_type)

    
    max_path = ("/v/raid1b/egibbons/MRIdata/DTI/noddi/max_values_%i_directions_raw.h5" %
                n_directions)
    maxs = readhdf5.read_hdf5(max_path,"max_values")[None,None,None,:]

    max_y_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/max_y_raw.h5"
    max_y = readhdf5.read_hdf5(max_y_path,"max_y")

    subsampling_pattern = subsampling.gensamples(n_directions,random_seed=random_seed)
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

def model_raw_new(data,n_directions,random_seed=400,loss_type="l1"):

    image_size = (128,128,n_directions)

    model = simple2d.res2dnew(image_size)
    model.compile(optimizer=Adam(lr=1e-3),
                  loss="mean_absolute_error",
                  metrics=["accuracy"])
    model.load_weights("/v/raid1b/egibbons/models/noddi-%i_new_raw.h5" %
                       (n_directions))
    print("2D dense model loaded for raw data.  Using %s loss" % loss_type)

    
    max_path = ("/v/raid1b/egibbons/MRIdata/DTI/noddi/max_values_%i_directions_raw.h5" %
                n_directions)
    maxs = readhdf5.read_hdf5(max_path,"max_values")[None,None,None,:]

    max_y_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/max_y_raw.h5"
    max_y = readhdf5.read_hdf5(max_y_path,"max_y")

    subsampling_pattern = subsampling.gensamples(n_directions,random_seed=random_seed)
    x = data[:,:,:,subsampling_pattern]
    x = x.transpose(2,0,1,3).astype(float)

    x /= maxs.squeeze()[None,None,None,:].astype(float)
    
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



def model_1d(data,n_directions,random_seed=400,loss_type="l1"):

    image_size = (128,128,n_directions)

    model = simple1d.res2d(image_size)
    model.compile(optimizer=Adam(lr=1e-3),
                  loss="mean_absolute_error",
                  metrics=["accuracy"])
    model.load_weights("/v/raid1b/egibbons/models/noddi-%i_1d_res_%s.h5" %
                       (n_directions, loss_type))
    print("2D dense model loaded.  Using %s loss" % loss_type)

    
    max_path = ("/v/raid1b/egibbons/MRIdata/DTI/noddi/max_values_%i_directions_2d.h5" %
                n_directions)
    maxs = readhdf5.read_hdf5(max_path,"max_values")[None,None,None,:]

    max_y_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/max_y_2d.h5"
    max_y = readhdf5.read_hdf5(max_y_path,"max_y")

    subsampling_pattern = subsampling.gensamples(n_directions,random_seed=random_seed)
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


def golkov_multi(data, n_directions, random_seed=400):

    max_path = ("/v/raid1b/egibbons/MRIdata/DTI/noddi/max_values_%i_directions_1d.h5" %
                n_directions)
    maxs = readhdf5.read_hdf5(max_path,"max_values")[None,None,None,:]

    max_y_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/max_y_1d.h5"
    max_y = readhdf5.read_hdf5(max_y_path,"max_y")
    
    subsampling_pattern = subsampling.gensamples(n_directions, random_seed=random_seed)

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

def sampling_2d(data,n_directions,random_seed=400,loss_type="l1"):

    image_size = (128,128,n_directions)

    model = simple2d.res2d(image_size)
    model.compile(optimizer=Adam(lr=1e-3),
                  loss="mean_absolute_error",
                  metrics=["accuracy"])
    model.load_weights("/v/raid1b/egibbons/models/noddi-%i_%i_seed_2d.h5" %
                       (n_directions, random_seed))
    print("2D dense model loaded.  Using %s loss" % loss_type)

    max_path = ("/v/raid1b/egibbons/MRIdata/DTI/noddi/max_values_%i_directions_%i_seed_2d.h5" %
                (n_directions, random_seed))
    maxs = readhdf5.read_hdf5(max_path,"max_values")[None,None,None,:]

    max_y_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/max_y_2d.h5"
    max_y = readhdf5.read_hdf5(max_y_path,"max_y")

    subsampling_pattern = subsampling.gensamples(n_directions,random_seed=random_seed)
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


def separate_2d(data,n_directions,random_seed=400,loss_type="l1",scaling=True):

    # load the data
    max_path = ("/v/raid1b/egibbons/MRIdata/DTI/noddi/max_values_%i_directions_2d.h5" %
                n_directions)
    maxs = readhdf5.read_hdf5(max_path,"max_values")[None,None,None,:]
    
    max_y_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/max_y_2d.h5"
    max_y = readhdf5.read_hdf5(max_y_path,"max_y")

    subsampling_pattern = subsampling.gensamples(n_directions,random_seed=random_seed)
    x = data[:,:,:,subsampling_pattern]
    x = x.transpose(2,0,1,3)

    x /= maxs.squeeze()[None,None,None,:]

    x_noddi = np.copy(x)
    x_gfa = np.copy(x)
    
    image_size = (128,128,n_directions)

    # noddi model
    model_noddi = simple2d_noddi.res2d(image_size)
    model_noddi.compile(optimizer=Adam(lr=1e-3),
                  loss="mean_absolute_error",
                  metrics=["accuracy"])
    model_noddi.load_weights("/v/raid1b/egibbons/models/noddi-%i_2d_noddi.h5" %
                             (n_directions))

    model_gfa = simple2d_gfa.res2d(image_size)
    model_gfa.compile(optimizer=Adam(lr=1e-3),
                  loss="mean_absolute_error",
                  metrics=["accuracy"])
    
    if scaling is True:
        model_gfa.load_weights("/v/raid1b/egibbons/models/noddi-%i_2d_gfa.h5" %
                               (n_directions))
        scaling_factor = 5
    else:
        print("no scaling")
        model_gfa.load_weights("/v/raid1b/egibbons/models/noddi-%i_2d_gfa_no_scale.h5" %
                               (n_directions))
        scaling_factor = 1
        
    print("2D dense model loaded.  Using %s loss" % loss_type)
    
    print("Predicting 2D separate...")
    start = time.time()
    prediction_noddi = model_noddi.predict(x_noddi, batch_size=10).transpose(1,2,0,3)
    prediction_gfa = model_gfa.predict(x_gfa, batch_size=10).transpose(1,2,0,3)
    print("Predictions completed...took: %f" % (time.time() - start))

    prediction = np.concatenate((prediction_noddi,
                                 prediction_gfa/scaling_factor),
                                axis=3)

    ### DISPLAY ###
    for ii in range(4):
        prediction[:,:,:,ii] *= max_y[ii]
    
        
    return prediction
