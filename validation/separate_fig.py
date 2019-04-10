import os
import sys
import time 

loss_type = "l1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
from matplotlib import pyplot as plt
from skimage import measure

sys.path.append("/home/mirl/egibbons/noddi")


from noddi_utils import noddistudy
from noddi_utils import predict
from noddi_utils import subsampling
from recon import imtools
from utils import readhdf5
from utils import display



### LOAD DATA ###
patient_number = "P080715"
noddi_data = noddistudy.NoddiData(patient_number)

max_y_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/max_y_2d.h5"
max_y = readhdf5.read_hdf5(max_y_path,"max_y")

data_full = noddi_data.get_full()
data_odi = noddi_data.get_odi()
data_fiso = noddi_data.get_fiso()
data_ficvf = noddi_data.get_ficvf()
data_gfa = noddi_data.get_gfa()

data_odi[data_odi>max_y[0]] = max_y[0]
data_fiso[data_fiso>max_y[1]] = max_y[1]
data_ficvf[data_ficvf>max_y[2]] = max_y[2]
data_gfa[data_gfa>max_y[3]] = max_y[3]

reference = np.concatenate((data_odi[:,:,:,None],
                            data_fiso[:,:,:,None],
                            data_ficvf[:,:,:,None],
                            data_gfa[:,:,:,None]),
                           axis=3).transpose(1,0,2,3)[::-1,::-1,:,:]

crop_factor = 100
roll_factor = -5
reference = np.roll(reference,roll_factor,axis=0)
reference = imtools.Crop(reference,crop_factor,crop_factor)

slice_use = 20

gfa_scale = 0.5
reference[:,:,:,3] /= gfa_scale


for ii in range(4):
    if ii == 0:
        montage_final = reference[:,:,slice_use,ii]
    else:
        montage_final = np.vstack((montage_final,
                                   reference[:,:,slice_use,ii]))

directions = [128, 64, 32, 24, 16]
models = ["2d", "separate_2d"]

cmap_use = "gray"
vmax_use = 1

n_directions = 24
for model_type in models:
    if model_type == "2d":
        prediction = predict.model_2d(data_full, n_directions).transpose(1,0,2,3)[::-1,::-1,:,:]
    else:
        prediction = predict.separate_2d(data_full, n_directions).transpose(1,0,2,3)[::-1,::-1,:,:]
        

    prediction[:,:,:,3] /= gfa_scale

    prediction = np.roll(prediction,roll_factor,axis=0)
    prediction = imtools.Crop(prediction,crop_factor,crop_factor)
    
    for ii in range(4):
        vmax_use = 1.
        
        montage_1 = np.hstack((prediction[:,:,slice_use,ii],
                               3*abs(prediction[:,:,slice_use,ii] -
                                   reference[:,:,slice_use,ii])))

            
        
        if ii == 0:
            montage_2 = montage_1
        else:
            montage_2 = np.vstack((montage_2,montage_1))

    montage_final = np.hstack((montage_final,montage_2))

plt.figure()
plt.imshow(abs(montage_final),cmap=cmap_use,vmax=vmax_use)
plt.colorbar()
plt.axis("off")
plt.savefig("../results/fig6_separate_fig.pdf", bbox_inches="tight",dpi=600)
plt.show()
        
