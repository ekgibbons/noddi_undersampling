import operator
import sys

from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np
from scipy import stats
from statsmodels.stats import multitest

sys.path.append("/home/mirl/egibbons/noddi")
from noddi_utils import metrics
from noddi_utils import noddistudy
from recon import imtools
from utils import display
from utils import readhdf5

def orient(data, index):

    rolls = [-14, -4, -5]
    crop_factor = [100, 100, 100]
    
    if len(data.shape) is not 4:
        data = data[:,:,:,None]

    data = data.transpose(1,0,2,3)[::-1,::-1].squeeze()
    
    return imtools.Crop(np.roll(data, rolls[index], axis=0),
                        crop_factor[index], crop_factor[index])


    
test_cases = ["P061114","P032315","P080715"]
slice_number = [26, 23, 22]

max_y_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/max_y_2d.h5"
max_y = readhdf5.read_hdf5(max_y_path,"max_y")

data = {}
data_types = ["odi", "fiso", "ficvf", "gfa"]
measurements = ["SSIM", "PSNR", "NRMSE"]

gfa_scale = 0.5

for ii, patient_number in enumerate(test_cases):
    
    noddi_data = noddistudy.NoddiData(patient_number)

    # data_odi = noddi_data.get_odi()
    # plt.figure()
    # display.render(orient(data_odi, ii))

    if ii is 3:
        continue
    
    
    path_data = ("/v/raid1b/egibbons/MRIdata/DTI/noddi/"
                 "processing/%s_24_directions_2d.h5" %
                 (patient_number))
    prediction = readhdf5.read_hdf5(path_data,"predictions")
    
    data_odi = noddi_data.get_odi()
    data_fiso = noddi_data.get_fiso()
    data_ficvf = noddi_data.get_ficvf()
    data_gfa = noddi_data.get_gfa()

    data_odi[data_odi>max_y[0]] = max_y[0]
    data_fiso[data_fiso>max_y[1]] = max_y[1]
    data_ficvf[data_ficvf>max_y[2]] = max_y[2]
    data_gfa[data_gfa>max_y[3]] = max_y[3]

    data_odi = orient(data_odi, ii)
    data_gfa = orient(data_gfa, ii)
    prediction = orient(prediction, ii)

    montage_row = np.hstack((data_odi[:,:,slice_number[ii]],
                             prediction[:,:,slice_number[ii],0],
                             data_gfa[:,:,slice_number[ii]]/gfa_scale,
                             prediction[:,:,slice_number[ii],3]/gfa_scale))

    if ii is 0:
        montage_final = montage_row
    else:
        montage_final = np.vstack((montage_final,
                                   montage_row))

plt.figure()
plt.imshow(abs(montage_final),cmap="gray",vmax=1.0)
plt.axis("off")
plt.savefig("../results/fig8_stroke.pdf", dpi=800, bbox_inches="tight")

plt.show()

    
