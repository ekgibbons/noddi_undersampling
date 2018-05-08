import sys

from matplotlib import pyplot as plt
import nibabel as nib
import numpy as np

sys.path.append("/home/mirl/egibbons/noddi")
from noddi_utils import noddistudy
from noddi_utils import predict
from utils import matutils
from utils import readhd5

patient_number = "P032315"
n_directions = 24
slice_use = 22

### LOAD DATA ###
path_mask = "/v/raid1b/egibbons/MRIdata/DTI/noddi/masks/P032315_mask.nii.gz"
mask = nib.load(path_mask)
mask = mask.get_data()

noddi_data = noddistudy.NoddiData(patient_number)

max_y_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/max_y_2d.h5"
max_y = readhd5.ReadHDF5(max_y_path,"max_y")

data_full = noddi_data.get_full()
data_odi = noddi_data.get_odi()*mask
data_fiso = noddi_data.get_fiso()*mask
data_ficvf = noddi_data.get_ficvf()*mask
data_gfa = noddi_data.get_gfa()*mask

data_odi[data_odi>max_y[0]] = max_y[0]
data_fiso[data_fiso>max_y[1]] = max_y[1]
data_ficvf[data_ficvf>max_y[2]] = max_y[2]
data_gfa[data_gfa>max_y[3]] = max_y[3]

reference = np.concatenate((data_odi[:,:,:,None],
                            data_fiso[:,:,:,None],
                            data_ficvf[:,:,:,None],
                            data_gfa[:,:,:,None]),
                           axis=3).transpose(1,0,2,3)[::-1,::-1,:,:]

for ii in range(4):
    if 0:
        cmap_use="hot"
    else:
        cmap_use="gray"
        
    plt.figure()
    plt.imshow(reference[:,:,slice_use,ii],cmap=cmap_use)
    plt.axis("off")

    plt.savefig("../results/fig3_reference_parameter_%i.pdf"
                % ii, bbox_inches="tight")



path_to_figs = ("/v/raid1b/khodgson/MRIdata/DTI/CNC_Imris/"
                "Stroke_patients/Stroke_DSI_Processing/"
                "Scripts/Processing_for_Eric/P032315")

data_types = ["odi", "fiso", "ficvf", "gfa"]

models = ["2d", "naive"]

for model_type in models:
    ii = 0

    if model_type == "2d":
        prediction = predict.model_2d(data_full,
                                      n_directions).transpose(1,0,2,3)[::-1,::-1,:,:]

    for data_type in data_types:
        if 0:
            cmap_use="hot"
        else:
            cmap_use="gray"
            
        if ii != 3:
            vmax_use = 1.
        else:
            vmax_use = np.amax(reference[:,:,slice_use,3])

        plt.figure()
            
        if model_type == "naive":    
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

            img_plot = np.concatenate((img_np[:,:,slice_use],
                                       abs(img_np[:,:,slice_use] -
                                           reference[:,:,slice_use,ii])),
                                      axis=1)
                
        else:
            img_plot = np.concatenate((prediction[:,:,slice_use,ii],
                                       abs(prediction[:,:,slice_use,ii] -
                                           reference[:,:,slice_use,ii])),
                                      axis=1)
        print(img_plot.shape)
        plt.imshow(img_plot,
                   cmap=cmap_use,
                   vmax=vmax_use)
        
        plt.colorbar()
        plt.axis("off")
        
        plt.savefig("../results/fig3_model_%s_parameter_%i.pdf" % (model_type, ii),
                    bbox_inches="tight")

        ii += 1


