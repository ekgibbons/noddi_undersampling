from __future__ import absolute_import

import h5py
import json
from matplotlib import pyplot as plt
import numpy as np
import os
import sys

sys.path.append("/home/mirl/egibbons/noddi")

from utils import display
from utils import mkdir



from noddi_utils import noddistudy

with open("../noddi_utils/noddi_metadata.json") as metadata:
    patient_database = json.load(metadata)

subsampling = [0, 1, 2, 3, 5, 6, 8, 10, 11, 12, 13, 15, 18, 21, 27, 31, 32,
               40, 41, 45, 47, 49, 52, 55, 57, 60, 63, 65, 66, 70,
               81, 82, 86, 94, 95, 99, 100, 102, 104, 107, 110, 113, 115, 118,
               123, 130, 135, 140, 145, 150, 155, 160, 164,
               167, 168, 174, 180, 184, 187, 190, 193, 197,
               200, 205]

num_cases = len(patient_database)
print("We have %i cases" % len(patient_database))
print("We have %i directions" % len(subsampling))

print("\n")
mm = 0 
for patient_number in sorted(patient_database.keys()):

    noddi_data = noddistudy.NoddiData(patient_number)

    if noddi_data.get_type() is "test":
        continue
    
    print("Currently reading: %s" % patient_number)

    data_full_temp = noddi_data.get_full()[:,:,2:(-1-3),:]
    data_odi_temp = noddi_data.get_odi()[:,:,2:(-1-3)]
    data_fiso_temp = noddi_data.get_fiso()[:,:,2:(-1-3)]
    data_ficvf_temp = noddi_data.get_ficvf()[:,:,2:(-1-3)]

    data_subsampled_temp = data_full_temp[:,:,:,subsampling]
    
    dim0, dim1, n_slices, n_channels = data_subsampled_temp.shape

    running_sum = 0
    for kk in range(n_slices):
        running_sum += np.sum(data_odi_temp[:,:,kk] > 1e-8)

    data_odi = np.zeros((running_sum,1))
    data_ficvf = np.zeros((running_sum,1))
    data_fiso = np.zeros((running_sum,1))
    data_subsampled = np.zeros((running_sum,n_channels))

    ll = 0

    for ii in range(dim0):
        for jj in range(dim1):
            for kk in range(n_slices):
                if data_odi_temp[ii,jj,kk] > 1e-8:
                    data_odi[ll,0] = data_odi_temp[ii,jj,kk]
                    data_fiso[ll,0] = data_fiso_temp[ii,jj,kk]
                    data_ficvf[ll,0] = data_ficvf_temp[ii,jj,kk]
                    data_subsampled[ll,:] = data_subsampled_temp[ii,jj,kk,:]
                    ll += 1

    assert running_sum == ll, \
        "The running_sum doesn't match the number of non-zero entries"
    
    if mm == 0:
        x = data_subsampled
        y_odi = data_odi
        y_fiso = data_fiso
        y_ficvf = data_ficvf
        
    else:
        x = np.concatenate((x,data_subsampled),
                           axis=0)
        y_odi = np.concatenate((y_odi,data_odi),
                               axis=0)
        y_fiso = np.concatenate((y_fiso,data_fiso),
                                axis=0)
        y_ficvf = np.concatenate((y_ficvf,data_ficvf),
                                 axis=0)

    mm += 1

max_values = np.amax(x,axis=0)
x /= max_values

hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/max_values_1d.h5","w")
hf.create_dataset("max_values", data=max_values)
hf.close


# hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/mean_values_1d.h5","w")
# hf.create_dataset("mean_values", data=mean_values)
# hf.close

# hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/std_values_1d.h5","w")
# hf.create_dataset("std_values", data=std_values)
# hf.close

hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/x_%i_directions_train_1d.h5" %
               len(subsampling),"w")
hf.create_dataset("x_%i_directions" % len(subsampling), data=x)
hf.close

hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/y_odi_train_1d.h5","w")
hf.create_dataset("y_odi",data=y_odi)
hf.close

hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/y_fiso_train_1d.h5","w")
hf.create_dataset("y_fiso",data=y_fiso)
hf.close

hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/y_ficvf_train_1d.h5","w")
hf.create_dataset("y_ficvf",data=y_ficvf)
hf.close

