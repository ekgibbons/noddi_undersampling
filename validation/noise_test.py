import os
import sys
import time 

loss_type = "l1"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
from matplotlib import pyplot as plt

sys.path.append("/home/mirl/egibbons/noddi")

from model_2d import dense2d
from model_2d import simple2d
from model_2d import unet2d
from noddi_utils import network_utils
from noddi_utils import noddistudy
from noddi_utils import predict
from noddi_utils import subsampling
from recon import imtools
from utils import readhd5
from utils import display

n_directions = 32

sampling_pattern = subsampling.gensamples(n_directions)

### LOAD DATA ###
patient_number = "P111816"
noddi_data = noddistudy.NoddiData(patient_number)

max_y_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/max_y_2d.h5"
max_y = readhd5.ReadHDF5(max_y_path,"max_y")

max_path = ("/v/raid1b/egibbons/MRIdata/DTI/noddi/max_values_%i_directions_2d.h5" %
            n_directions)
maxs = readhd5.ReadHDF5(max_path,"max_values")[None,None,None,:]

data_full = noddi_data.get_full()
data_odi = noddi_data.get_odi()
data_fiso = noddi_data.get_fiso()
data_ficvf = noddi_data.get_ficvf()
data_gfa = noddi_data.get_gfa()

data_odi[data_odi>max_y[0]] = max_y[0]
data_fiso[data_fiso>max_y[1]] = max_y[1]
data_ficvf[data_ficvf>max_y[2]] = max_y[2]
data_gfa[data_gfa>max_y[3]] = max_y[3]

rician_level = [1, 3, 5, 7]
slice_use = 25

for s in rician_level:
    data_temp = imtools.AddRician(data_full, s)
    
    prediction_1d = predict.golkov_multi(data_temp, n_directions)
    prediction_2d = predict.model_2d(data_temp, n_directions)
    
    plt.figure()
    plt.imshow(data_temp[:,:,slice_use,10])

    montage_1 = np.concatenate((data_odi[:,:,slice_use],
                                data_fiso[:,:,slice_use],
                                data_ficvf[:,:,slice_use],
                                data_gfa[:,:,slice_use]/0.5),
                             axis=1)
    
    montage_2 = np.concatenate((prediction_1d[:,:,slice_use,0],
                                prediction_1d[:,:,slice_use,1],
                                prediction_1d[:,:,slice_use,2],
                                prediction_1d[:,:,slice_use,3]/0.5),
                               axis=1)

    montage_3 = np.concatenate((prediction_2d[:,:,slice_use,0],
                                prediction_2d[:,:,slice_use,1],
                                prediction_2d[:,:,slice_use,2],
                                prediction_2d[:,:,slice_use,3]/0.5),
                               axis=1)
    
    montage_combine = np.concatenate((montage_1,
                                      montage_2,
                                      abs(montage_1 - montage_2),
                                      montage_3,
                                      abs(montage_1 - montage_3)),
                                     axis=0)

    plt.figure()
    plt.imshow(montage_combine)
    plt.title("ODI, FISO, FICVF, GFA")
    plt.axis("off")


    
    plt.show()
    
    
