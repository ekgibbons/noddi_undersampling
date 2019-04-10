import os
import sys
import time 

loss_type = "l1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
from matplotlib import pyplot as plt
from skimage import measure

sys.path.append("/home/mirl/egibbons/noddi")

from model_2d import dense2d
from model_2d import simple2d
from model_2d import unet2d
from noddi_utils import network_utils
from noddi_utils import noddistudy
from noddi_utils import predict
from noddi_utils import subsampling
from recon import imtools
from utils import readhdf5
from utils import display

test_cases = ["P032315","P080715","P061114",
              "N011118A","N011118B"]

n_directions = 24

sampling_pattern = subsampling.gensamples(n_directions, random_seed=425)

### LOAD DATA ###
patient_number = test_cases[0]
noddi_data = noddistudy.NoddiData(patient_number)

# max_y_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/max_y_2d.h5"
# max_y = readhdf5.read_hdf5(max_y_path,"max_y")

data_full = noddi_data.get_full()
data_raw = noddi_data.get_raw()
data_odi = noddi_data.get_odi().transpose(1,0,2)[::-1,::-1]
data_fiso = noddi_data.get_fiso().transpose(1,0,2)[::-1,::-1]
data_ficvf = noddi_data.get_ficvf().transpose(1,0,2)[::-1,::-1]
data_gfa = noddi_data.get_gfa().transpose(1,0,2)[::-1,::-1]

# data_odi[data_odi>max_y[0]] = max_y[0]
# data_fiso[data_fiso>max_y[1]] = max_y[1]
# data_ficvf[data_ficvf>max_y[2]] = max_y[2]
# data_gfa[data_gfa>max_y[3]] = max_y[3]

slice_use = 22
 
prediction_1d = predict.golkov_multi(data_full,
                                 n_directions).transpose(1,0,2,3)[::-1,::-1]
prediction_2d = predict.model_2d(data_full,
                                 n_directions).transpose(1,0,2,3)[::-1,::-1]
prediction_raw = predict.model_raw(data_raw,
                                   n_directions).transpose(1,0,2,3)[::-1,::-1]
prediction_raw_new = predict.model_raw_new(data_raw,
                                           n_directions).transpose(1,0,2,3)[::-1,::-1]


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

montage_4 = np.concatenate((prediction_raw[:,:,slice_use,0],
                            prediction_raw[:,:,slice_use,1],
                            prediction_raw[:,:,slice_use,2],
                            prediction_raw[:,:,slice_use,3]/0.5),
                           axis=1)

montage_5 = np.concatenate((prediction_raw_new[:,:,slice_use,0],
                            prediction_raw_new[:,:,slice_use,1],
                            prediction_raw_new[:,:,slice_use,2],
                            prediction_raw_new[:,:,slice_use,3]/0.5),
                           axis=1)



montage_combine = np.concatenate((montage_1,
                                  montage_2,
                                  abs(montage_1 - montage_2),
                                  montage_3,
                                  abs(montage_1 - montage_3),
                                  montage_4,
                                  abs(montage_1 - montage_4),
                                  montage_5,
                                  abs(montage_1 - montage_5)),
                                 axis=0)

plt.figure()
plt.imshow(montage_combine,cmap="gray")
plt.title("ODI, FISO, FICVF, GFA")
plt.axis("off")

plt.show()
    
    
