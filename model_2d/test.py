import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from matplotlib import pyplot as plt
import numpy as np

sys.path.append("/home/mirl/egibbons/noddi")

from noddi_utils import noddistudy
from noddi_utils import subsampling
from noddi_utils import predict
from utils import display
from utils import readhd5

loss_type = "l1"

n_channels = 16

### LOAD DATA ###
patient_number = "N011618"
noddi_data = noddistudy.NoddiData(patient_number)

max_y_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/max_y_2d.h5"
max_y = readhd5.ReadHDF5(max_y_path,"max_y")

data_full = noddi_data.get_full()
data_odi = noddi_data.get_odi()
data_fiso = noddi_data.get_fiso()
data_ficvf = noddi_data.get_ficvf()
data_gfa = noddi_data.get_gfa()
data_md = noddi_data.get_md()
data_ad = noddi_data.get_ad()
data_fa = noddi_data.get_fa()

data_odi[data_odi>max_y[0]] = max_y[0]
data_fiso[data_fiso>max_y[1]] = max_y[1]
data_ficvf[data_ficvf>max_y[2]] = max_y[2]
data_gfa[data_gfa>max_y[3]] = max_y[3]
data_md[data_md>max_y[4]] = max_y[4]
data_ad[data_ad>max_y[5]] = max_y[5]
data_fa[data_fa>max_y[6]] = max_y[6]

prediction = predict.model_2d(data_full, n_channels)

slice_use = 25

montage_top = np.concatenate((data_odi[:,:,slice_use],
                              data_fiso[:,:,slice_use],
                              data_ficvf[:,:,slice_use]),
                             axis=1)

montage_bottom = np.concatenate((prediction[:,:,slice_use,0],
                                 prediction[:,:,slice_use,1],
                                 prediction[:,:,slice_use,2]),
                                axis=1)

montage_combine = np.concatenate((montage_top,
                                  montage_bottom,
                                  abs(montage_top - montage_bottom)),
                                 axis=0)

plt.figure()
plt.imshow(montage_combine)
plt.title("ODI, FISO, FICVF")
plt.axis("off")

plt.figure()
plt.imshow(np.concatenate((data_gfa[:,:,slice_use],
                           prediction[:,:,slice_use,3],
                           abs(data_gfa[:,:,slice_use] -
                               prediction[:,:,slice_use,3])),
                          axis=0), vmax=0.5)
plt.title("GFA")
plt.axis("off")


# plt.figure()
# plt.imshow(np.concatenate((data_md[:,:,slice_use],
#                            prediction[:,:,slice_use,4],
#                            abs(data_md[:,:,slice_use] - prediction[:,:,slice_use,4])),
#                           axis=0), vmax=4e-3)
# plt.title("MD")
# plt.axis("off")

# plt.figure()
# plt.imshow(np.concatenate((data_ad[:,:,slice_use],
#                            prediction[:,:,slice_use,5],
#                            abs(data_ad[:,:,slice_use] - prediction[:,:,slice_use,5])),
#                           axis=0), vmax=4e-3)
# plt.title("AD")
# plt.axis("off")

# plt.figure()
# plt.imshow(np.concatenate((data_fa[:,:,slice_use],
#                            prediction[:,:,slice_use,6],
#                            abs(data_fa[:,:,slice_use] - prediction[:,:,slice_use,6])),
#                           axis=0), vmax=0.7)
# plt.title("FA")
# plt.axis("off")

plt.show()






