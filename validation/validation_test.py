import sys

from matplotlib as pyplot as plt
import numpy as np

sys.path.append("/home/mirl/egibbons/noddi")
from noddi_utils import metrics
from noddi_utils import predict
from utils import display
from utils import readhd5

n_directions = 64

max_y_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/max_y_2d.h5"
max_y = readhd5.ReadHDF5(max_y_path,"max_y")

data_full = noddi_data.get_full()
data_odi = noddi_data.get_odi()
data_fiso = noddi_data.get_fiso()
data_ficvf = noddi_data.get_ficvf()
data_gfa = noddi_data.get_gfa()

data_odi[data_odi>max_y[0]] = max_y[0]
data_fiso[data_fiso>max_y[1]] = max_y[1]
data_ficvf[data_ficvf>max_y[2]] = max_y[2]
data_gfa[data_gfa>max_y[3]] = max_y[3]

print(data_gfa.shape)

# prediction_1d = predict.predict_1d(data_full, n_channels)
# prediction_2d = predict.predict_2d(data_full, n_channels)

# gfa_bins_2d, gfa_locations, _ = metric.get_delta_histogram(prediction_1d[:,:,:,3],
#                                                            data_gfa[
