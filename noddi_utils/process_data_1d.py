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

### SETUP ###
with open("noddi_metadata.json") as metadata:
    patient_database = json.load(metadata)
print("We have %i cases" % len(patient_database))

directions = [128, 64, 32, 24, 16, 8]

### MAIN LOOP ###

for num_directions in directions:
    print("Generating 1D data for %i directions" % num_directions)

    subsampling_indices = subsampling.gensamples(num_directions)

    num_cases = len(patient_database)
    
    print("\n")
    mm = 0 
    for patient_number in sorted(patient_database.keys()):

        noddi_data = noddistudy.NoddiData(patient_number)

        if (noddi_data.get_type() == "test" or
            noddi_data.get_type() == "duplicate"):
            continue
    
        print("Currently reading: %s as %s data" %
              (patient_number, noddi_data.get_type()))

        data_full = noddi_data.get_full()[:,:,2:(-1-3),:]
        data_odi = noddi_data.get_odi()[:,:,2:(-1-3)]
        data_fiso = noddi_data.get_fiso()[:,:,2:(-1-3)]
        data_ficvf = noddi_data.get_ficvf()[:,:,2:(-1-3)]
        data_gfa = noddi_data.get_gfa()[:,:,2:(-1-3)]

        data_subsampled = data_full[:,:,:,subsampling_indices]
        
        dim0, dim1, n_slices, n_channels = data_subsampled.shape
        
        running_sum = data_odi.size
        
        data_odi = data_odi.reshape(-1,1)
        data_fiso = data_fiso.reshape(-1,1)
        data_ficvf = data_ficvf.reshape(-1,1)
        data_gfa = data_gfa.reshape(-1,1)
        data_subsampled = data_subsampled.transpose(3,0,1,2)
        data_subsampled = data_subsampled.reshape(n_channels,-1)
        data_subsampled = data_subsampled.transpose(1,0)

        if mm == 0:
            x = data_subsampled
            y_odi = data_odi
            y_fiso = data_fiso
            y_ficvf = data_ficvf
            y_gfa = data_gfa
        
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
        mm += 1

    print("Final data shape: (%i, %i)" % x.shape)
    
    max_values = np.amax(x,axis=0)
    x /= max_values
    
    max_y = np.array([1.0, 1.0, 1.0, 0.8])
    
    y_odi[y_odi>max_y[0]] = max_y[0]
    y_fiso[y_fiso>max_y[1]] = max_y[1]
    y_ficvf[y_ficvf>max_y[2]] = max_y[2]
    y_gfa[y_gfa>max_y[3]] = max_y[3]
    
    y_odi /= max_y[0]
    y_fiso /= max_y[1]
    y_ficvf /= max_y[2]
    y_gfa /= max_y[3]
    
    hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/max_values_%i_directions_1d.h5"
                   % num_directions,"w")
    hf.create_dataset("max_values", data=max_values)
    hf.close
    
    hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/max_y_1d.h5","w")
    hf.create_dataset("max_y", data=max_y)
    hf.close
    
    hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/x_%i_directions_1d.h5" %
                   num_directions,"w")
    hf.create_dataset("x_%i_directions" % num_directions, data=x)
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
