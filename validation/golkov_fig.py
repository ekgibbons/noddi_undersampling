import os
import sys
import time 

loss_type = "l1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
from matplotlib import pyplot as plt
import nibabel as nib
from skimage import measure

sys.path.append("/home/mirl/egibbons/noddi")


from noddi_utils import noddistudy
from noddi_utils import predict
from noddi_utils import subsampling
from recon import imtools
from utils import display
from utils import matutils
from utils import readhdf5

### LOAD DATA ###
patient_number = "P032315"
noddi_data = noddistudy.NoddiData(patient_number)

path_to_figs = ("/v/raid1b/khodgson/MRIdata/DTI/CNC_Imris/"
                "Stroke_patients/Stroke_DSI_Processing/"
                "Scripts/Processing_for_Eric/P032315")

data_types = ["odi", "fiso", "ficvf", "gfa"]


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
roll_factor = -4
reference = np.roll(reference,roll_factor,axis=0)
reference = imtools.Crop(reference,crop_factor,crop_factor)


slice_use = 21

gfa_scale = 0.5 #np.amax(reference[:,:,:,3])
reference[:,:,:,3] /= gfa_scale


for ii in range(4):
    if ii == 0:
        montage_final = reference[:,:,slice_use,ii]
    else:
        montage_final = np.vstack((montage_final,
                                   reference[:,:,slice_use,ii]))


        
directions = [128, 64, 32, 24, 16]
models = ["2d", "1d", "naive"]

cmap_use = "gray"
n_directions = 24
for model_type in models:
    if model_type == "naive":
        for ii, data_type in enumerate(data_types):
            if ii != 3:
                path_figure = "%s/%s_hydi_%s.nii" % (path_to_figs,
                                                     patient_number,
                                                     data_type)
                img = nib.load(path_figure)
                img_np = img.get_data().transpose(1,0,2)[::-1,::-1,:]
                
            else:
                path_figure = "%s/GFA.mat" % (path_to_figs)
                
                img_np = matutils.MatReader(path_figure,
                                            keyName="GFA").transpose(1,0,2)[::-1,::-1,:]
                
            if ii == 0:
                prediction = img_np[:,:,:,None]
            else:
                prediction = np.concatenate((prediction, img_np[:,:,:,None]),
                                            axis=3)


        
    elif model_type == "2d":
        prediction = predict.model_2d(data_full, n_directions).transpose(1,0,2,3)[::-1,::-1,:,:]
    elif model_type == "1d":
        prediction = predict.golkov_multi(data_full, n_directions).transpose(1,0,2,3)[::-1,::-1,:,:]

    prediction = np.roll(prediction,roll_factor,axis=0)
    prediction = imtools.Crop(prediction,crop_factor,crop_factor)

        
    prediction[:,:,:,3] /= gfa_scale
        
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

    if model_type == "2d":
        im_2d_1 = prediction[30:70,30:70,slice_use,0]
        im_2d_2 = prediction[40:80,45:85,slice_use,3]
    elif model_type == "1d":
        im_1d_1 = prediction[30:70,30:70,slice_use,0]
        im_1d_2 = prediction[40:80,45:85,slice_use,3]
    else:
        im_na_1 = prediction[30:70,30:70,slice_use,0]
        im_na_2 = prediction[40:80,45:85,slice_use,3]

reference_1 = reference[30:70,30:70,slice_use,0]
reference_2 = reference[40:80,45:85,slice_use,3]
        
montage_zoom_1 = np.hstack((reference_1,im_2d_1,abs(reference_1-im_2d_1),
                            im_1d_1,abs(reference_1-im_1d_1),
                            im_na_1,abs(reference_1-im_na_1)))
montage_zoom_2 = np.hstack((reference_2,im_2d_2,abs(reference_2-im_2d_2),
                            im_1d_2,abs(reference_2-im_1d_2),
                            im_na_2,abs(reference_2-im_na_2)))
montage_zoom = np.vstack((montage_zoom_1,montage_zoom_2))        


    
plt.figure()
plt.imshow(abs(montage_final),cmap=cmap_use,vmax=vmax_use)
plt.colorbar()
plt.axis("off")
plt.savefig("../results/fig4_directions_fig.pdf", bbox_inches="tight",dpi=600)

plt.figure()
plt.imshow(abs(montage_zoom),cmap=cmap_use,vmax=vmax_use)
plt.colorbar()
plt.axis("off")
plt.savefig("../results/fig4_zoom_fig.pdf", bbox_inches="tight",dpi=600)


    






plt.show()
