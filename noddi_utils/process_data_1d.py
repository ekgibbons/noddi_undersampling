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
from noddi_utils import subsampling

with open("../noddi_utils/noddi_metadata.json") as metadata:
    patient_database = json.load(metadata)

num_directions = 16
print("Generating 1D data for %i directions" % num_directions)
subsampling = subsampling.gensamples(num_directions, shuffle=False)
# print(subsampling)

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
    data_gfa_temp = noddi_data.get_gfa()[:,:,2:(-1-3)]
    data_md_temp = noddi_data.get_md()[:,:,2:(-1-3)]
    data_ad_temp = noddi_data.get_ad()[:,:,2:(-1-3)]
    data_fa_temp = noddi_data.get_fa()[:,:,2:(-1-3)]

    data_subsampled_temp = data_full_temp[:,:,:,subsampling]
    
    dim0, dim1, n_slices, n_channels = data_subsampled_temp.shape

    # running_sum = 0
    # for kk in range(n_slices):
    #     running_sum += np.sum(data_odi_temp[:,:,kk] > 1e-8)

    running_sum = data_odi_temp.size
        
    data_odi = np.zeros((running_sum,1))
    data_ficvf = np.zeros((running_sum,1))
    data_fiso = np.zeros((running_sum,1))
    data_gfa = np.zeros((running_sum,1))
    data_md = np.zeros((running_sum,1))
    data_ad = np.zeros((running_sum,1))
    data_fa = np.zeros((running_sum,1))
    data_subsampled = np.zeros((running_sum,n_channels))
    
    ll = 0

    for ii in range(dim0):
        for jj in range(dim1):
            for kk in range(n_slices):
                # if data_odi_temp[ii,jj,kk] > 1e-8:
                if True:
                    data_odi[ll,0] = data_odi_temp[ii,jj,kk]
                    data_fiso[ll,0] = data_fiso_temp[ii,jj,kk]
                    data_ficvf[ll,0] = data_ficvf_temp[ii,jj,kk]
                    data_gfa[ll,0] = data_gfa_temp[ii,jj,kk]
                    data_md[ll,0] = data_md_temp[ii,jj,kk]
                    data_ad[ll,0] = data_ad_temp[ii,jj,kk]
                    data_fa[ll,0] = data_fa_temp[ii,jj,kk]
                    data_subsampled[ll,:] = data_subsampled_temp[ii,jj,kk,:]
                    ll += 1

    assert running_sum == ll, \
        "The running_sum doesn't match the number of non-zero entries"
    
    if mm == 0:
        x = data_subsampled
        y_odi = data_odi
        y_fiso = data_fiso
        y_ficvf = data_ficvf
        y_gfa = data_gfa
        y_md = data_md
        y_ad = data_ad
        y_fa = data_fa
        
    else:
        x = np.concatenate((x,data_subsampled),
                           axis=0)
        y_odi = np.concatenate((y_odi,data_odi),
                               axis=0)
        y_fiso = np.concatenate((y_fiso,data_fiso),
                                axis=0)
        y_ficvf = np.concatenate((y_ficvf,data_ficvf),
                                 axis=0)
        y_gfa = np.concatenate((y_gfa,data_gfa),
                                 axis=0)
        y_md = np.concatenate((y_md,data_md),
                                 axis=0)
        y_ad = np.concatenate((y_ad,data_ad),
                                 axis=0)
        y_fa = np.concatenate((y_fa,data_fa),
                                 axis=0)

    mm += 1

max_values = np.amax(x,axis=0)
x /= max_values

max_y = np.array([1.0, 1.0, 1.0,
                  0.8, 1e-2, 1e-2, 0.8])

y_odi[y_odi>max_y[0]] = max_y[0]
y_fiso[y_fiso>max_y[1]] = max_y[1]
y_ficvf[y_ficvf>max_y[2]] = max_y[2]
y_gfa[y_gfa>max_y[3]] = max_y[3]
y_md[y_md>max_y[4]] = max_y[4]
y_ad[y_ad>max_y[5]] = max_y[5]
y_fa[y_fa>max_y[6]] = max_y[6]

y_odi /= max_y[0]
y_fiso /= max_y[1]
y_ficvf /= max_y[2]
y_gfa /= max_y[3]
y_md /= max_y[4]
y_ad /= max_y[5]
y_fa /= max_y[6]

hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/max_values_1d.h5","w")
hf.create_dataset("max_values", data=max_values)
hf.close

hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/max_y_1d.h5","w")
hf.create_dataset("max_y", data=max_y)
hf.close

hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/x_%i_directions_1d.h5" %
               len(subsampling),"w")
hf.create_dataset("x_%i_directions" % len(subsampling), data=x)
hf.close

hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/y_odi_1d.h5","w")
hf.create_dataset("y_odi",data=y_odi)
hf.close

hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/y_fiso_1d.h5","w")
hf.create_dataset("y_fiso",data=y_fiso)
hf.close

hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/y_ficvf_1d.h5","w")
hf.create_dataset("y_ficvf",data=y_ficvf)
hf.close

hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/y_gfa_1d.h5","w")
hf.create_dataset("y_gfa",data=y_gfa)
hf.close

hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/y_md_1d.h5","w")
hf.create_dataset("y_md",data=y_md)
hf.close

hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/y_ad_1d.h5","w")
hf.create_dataset("y_ad",data=y_ad)
hf.close

hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/y_fa_1d.h5","w")
hf.create_dataset("y_fa",data=y_fa)
hf.close


