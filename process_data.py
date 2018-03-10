import noddistudy
import json
import h5py

import numpy as np

from utils import mkdir


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
for patient_number in patient_database.keys():
    print("Currently reading: %s" % patient_number)
    
    noddi_data = noddistudy.NoddiData(patient_number)

    data_full = noddi_data.get_full()
    data_odi = noddi_data.get_odi()
    data_fiso = noddi_data.get_fiso()
    data_subsampled = data_full[:,:,:,subsampling]

    if ii == 0:
        x = data_subsampled.transpose(0,1,3,2)
        y_odi = data_odi[:,:,:,None].transpose(0,1,3,2)
        y_fiso = data_fiso[:,:,:,None].transpose(0,1,3,2)
        
    else:
        x = np.concatenate((x, data_subsampled.transpose(0,1,3,2)),
                           axis=3)
        y_odi = np.concatenate((y_odi, data_odi[:,:,:,None].transpose(0,1,3,2)),
                               axis=3)
        y_fiso = np.concatenate((y_fiso, data_fiso[:,:,:,None].transpose(0,1,3,2)),
                                axis=3)

    ii += 1

    if (ii > 32):
        break

hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/x_%i_directions.h5" %
               len(subsampling),"w")
hf.create_dataset("x_%i_directions" % len(subsampling), data=x)
hf.close

hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/y_odi.h5","w")
hf.create_dataset("y_odi",data=y_odi)
hf.close

hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/y_fiso.h5","w")
hf.create_dataset("y_fiso",data=y_fiso)
hf.close

