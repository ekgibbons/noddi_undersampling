import sys

from matplotlib import pyplot as plt
import nibabel as nib
import numpy as np

sys.path.append("/home/mirl/egibbons/noddi")
from noddi_utils import noddistudy
from recon import imtools
from utils import matutils
from utils import readhd5

patient_number = "P080715"

slice_use = 20

directions = [128, 64, 32, 24, 16, 8]

max_y_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/max_y_2d.h5"
max_y = readhd5.ReadHDF5(max_y_path,"max_y")

data = {}
models = ["2d", "separate_no_scale_2d"]

noddi_data = noddistudy.NoddiData(patient_number)
data_gfa = noddi_data.get_gfa()[:,:,slice_use].transpose(1,0)[::-1,::-1]
data_gfa[data_gfa>max_y[3]] = max_y[3]

crop_factor = 100
roll_factor = -5
data_gfa = np.roll(data_gfa,roll_factor,axis=0)
data_gfa = imtools.Crop(data_gfa,crop_factor,crop_factor)
data_gfa /= 0.5

range_gfa = np.amax(data_gfa) - np.amin(data_gfa)

jj = 0
for model_type in models:

    ii = 0
    for n_directions in directions:
        path_data = ("/v/raid1b/egibbons/MRIdata/DTI/noddi/"
                     "processing/%s_%i_directions_%s.h5" %
                     (patient_number, n_directions, model_type))
        
        data = readhd5.ReadHDF5(path_data, "predictions")[:,:,slice_use,3].transpose(1,0)[::-1,::-1]

        data = np.roll(data,roll_factor,axis=0)
        data = imtools.Crop(data,crop_factor,crop_factor)
        data /= 0.5
        
        data_gfa[data_gfa>max_y[3]] = max_y[3]

        difference_image = 3*abs(data_gfa - data)

        print("model: %s, directions: %i" % (model_type, n_directions))
        
        if ii is 0:
            top_montage = data
            bottom_montage = difference_image
        else:
            top_montage = np.hstack((top_montage, data))
            bottom_montage = np.hstack((bottom_montage, difference_image))

        ii += 1

    montage_model = np.vstack((top_montage, bottom_montage))

    if jj is 0:
        montage_final = montage_model
    else:
        montage_final = np.vstack((montage_final, montage_model))

    jj += 1

plt.figure()
plt.imshow(abs(montage_final),cmap="gray",vmax=1)
plt.colorbar()
plt.axis("off")
plt.savefig("../results/fig6_gfa_comp.pdf",bbox_inches="tight",dpi=500)
plt.show()
