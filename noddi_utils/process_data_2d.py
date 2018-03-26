import json
import h5py

from matplotlib import pyplot as plt
import numpy as np

from utils import display
from utils import mkdir

import noddistudy

with open("noddi_metadata.json") as metadata:
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
ii = 0 
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
    
    for jj in range(8):

        if jj == 0:
            data_odi = data_odi_temp
            data_fiso = data_fiso_temp
            data_ficvf = data_ficvf_temp
            data_subsampled = data_subsampled_temp

        else:
            data_odi_temp = np.rot90(data_odi_temp,1,(0,1))
            data_fiso_temp = np.rot90(data_fiso_temp,1,(0,1))
            data_ficvf_temp = np.rot90(data_ficvf_temp,1,(0,1))
            data_subsampled_temp = np.rot90(data_subsampled_temp,1,(0,1))

            data_odi = np.concatenate((data_odi,
                                       data_odi_temp),
                                      axis=2)
            data_fiso = np.concatenate((data_fiso,
                                        data_fiso_temp),
                                       axis=2)
            data_ficvf = np.concatenate((data_ficvf,
                                         data_ficvf_temp),
                                        axis=2)
            data_subsampled = np.concatenate((data_subsampled,
                                              data_subsampled_temp),
                                             axis=2)
            
        if jj == 3:
            data_odi_temp = data_odi_temp[::-1,:,:]
            data_fiso_temp = data_fiso_temp[::-1,:,:]
            data_ficvf_temp = data_ficvf_temp[::-1,:,:]
            data_subsampled_temp = data_subsampled_temp[::-1,:,:,:]


    data_subsampled /= np.amax(data_subsampled)
    data_odi /= np.amax(data_odi)
    data_fiso /= np.amax(data_fiso)
    data_ficvf /= np.amax(data_ficvf)
    
    if ii == 0:
        x = data_subsampled.transpose(0,1,3,2)
        y_odi = data_odi[:,:,:,None].transpose(0,1,3,2)
        y_fiso = data_fiso[:,:,:,None].transpose(0,1,3,2)
        y_ficvf = data_ficvf[:,:,:,None].transpose(0,1,3,2)
        
    else:
        x = np.concatenate((x,
                            data_subsampled.transpose(0,1,3,2)),
                           axis=3)
        y_odi = np.concatenate((y_odi,
                                data_odi[:,:,:,None].transpose(0,1,3,2)),
                               axis=3)
        y_fiso = np.concatenate((y_fiso,
                                 data_fiso[:,:,:,None].transpose(0,1,3,2)),
                                axis=3)
        y_ficvf = np.concatenate((y_ficvf,
                                  data_ficvf[:,:,:,None].transpose(0,1,3,2)),
                                 axis=3)

    ii += 1

hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/x_%i_directions_train.h5" %
               len(subsampling),"w")
hf.create_dataset("x_%i_directions" % len(subsampling), data=x)
hf.close

hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/y_odi_train.h5","w")
hf.create_dataset("y_odi",data=y_odi)
hf.close

hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/y_fiso_train.h5","w")
hf.create_dataset("y_fiso",data=y_fiso)
hf.close

hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/y_ficvf_train.h5","w")
hf.create_dataset("y_ficvf",data=y_ficvf)
hf.close

