import h5py
import json
import sys

from matplotlib import pyplot as plt
import numpy as np
from skimage.util import shape 

sys.path.append("/home/mirl/egibbons/noddi")

from noddi_utils import noddistudy
from noddi_utils import subsampling
from utils import display
from utils import mkdir

def create_patches(data, window_shape, overlap=0.5):

    step = int(window_shape[0]*overlap)
    
    data_patches = shape.view_as_windows(data,
                                         window_shape=window_shape,
                                         step=step)

    data_patches = data_patches.transpose(4,5,7,6,0,1,2,3).squeeze()
    data_patches = data_patches.reshape(window_shape[0],window_shape[1],
                                        window_shape[3],-1)
    data_patches = data_patches.transpose(0,1,3,2)

    return data_patches

### SETUP ###
with open("noddi_metadata.json") as metadata:
    patient_database = json.load(metadata)

subsampling = subsampling.gensamples(16)
print(subsampling)


num_cases = len(patient_database)
print("We have %i cases" % len(patient_database))
print("We have %i directions" % len(subsampling))

### DATA LOADING AND AUGMENTATION ###
print("\n")
ii = 0 
for patient_number in sorted(patient_database.keys()):

    noddi_data = noddistudy.NoddiData(patient_number)

    if noddi_data.get_type() is "test":
        continue
    
    print("Currently reading: %s" % patient_number)


    data_full_temp = noddi_data.get_full()[:,:,2:(-1-3),:]
    data_odi_temp = noddi_data.get_odi()[:,:,2:(-1-3),None]
    data_fiso_temp = noddi_data.get_fiso()[:,:,2:(-1-3),None]
    data_ficvf_temp = noddi_data.get_ficvf()[:,:,2:(-1-3),None]
    data_gfa_temp = noddi_data.get_gfa()[:,:,2:(-1-3),None]
    data_md_temp = noddi_data.get_md()[:,:,2:(-1-3),None]
    data_ad_temp = noddi_data.get_ad()[:,:,2:(-1-3),None]
    data_fa_temp = noddi_data.get_fa()[:,:,2:(-1-3),None]
    
    data_subsampled_temp = data_full_temp[:,:,:,subsampling]

    dim0, dim1, n_slices, n_channels = data_subsampled_temp.shape

    data_window_shape = (32,32,n_slices,n_channels)
    data_subsampled_temp = create_patches(data_subsampled_temp,
                                          data_window_shape)

    parameter_window_shape = (32,32,n_slices,1)
    data_odi_temp = create_patches(data_odi_temp,
                                   parameter_window_shape)

    data_fiso_temp = create_patches(data_fiso_temp,
                                   parameter_window_shape)
    
    data_ficvf_temp = create_patches(data_ficvf_temp,
                                   parameter_window_shape)

    data_gfa_temp = create_patches(data_gfa_temp,
                                   parameter_window_shape)

    data_ad_temp = create_patches(data_ad_temp,
                                   parameter_window_shape)

    data_md_temp = create_patches(data_md_temp,
                                   parameter_window_shape)

    data_fa_temp = create_patches(data_fa_temp,
                                   parameter_window_shape)

    for jj in range(1):

        if jj == 0:
            data_odi = data_odi_temp
            data_fiso = data_fiso_temp
            data_ficvf = data_ficvf_temp
            data_gfa = data_gfa_temp
            data_md = data_md_temp
            data_ad = data_ad_temp
            data_fa = data_fa_temp
            data_subsampled = data_subsampled_temp

        else:
            data_odi_temp = np.rot90(data_odi_temp,1,(0,1))
            data_fiso_temp = np.rot90(data_fiso_temp,1,(0,1))
            data_ficvf_temp = np.rot90(data_ficvf_temp,1,(0,1))
            data_gfa_temp = np.rot90(data_gfa_temp,1,(0,1))
            data_md_temp = np.rot90(data_md_temp,1,(0,1))
            data_ad_temp = np.rot90(data_ad_temp,1,(0,1))
            data_fa_temp = np.rot90(data_fa_temp,1,(0,1))
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
            data_gfa = np.concatenate((data_gfa,
                                       data_gfa_temp),
                                      axis=2)
            data_md = np.concatenate((data_md,
                                      data_md_temp),
                                     axis=2)
            data_ad = np.concatenate((data_ad,
                                       data_ad_temp),
                                      axis=2)
            data_fa = np.concatenate((data_fa,
                                      data_fa_temp),
                                     axis=2)
            data_subsampled = np.concatenate((data_subsampled,
                                              data_subsampled_temp),
                                             axis=2)
            
        if jj == 3:
            data_odi_temp = data_odi_temp[::-1,:,:]
            data_fiso_temp = data_fiso_temp[::-1,:,:]
            data_ficvf_temp = data_ficvf_temp[::-1,:,:]
            data_gfa_temp = data_gfa_temp[::-1,:,:]
            data_md_temp = data_md_temp[::-1,:,:]
            data_ad_temp = data_ad_temp[::-1,:,:]
            data_fa_temp = data_fa_temp[::-1,:,:]
            data_subsampled_temp = data_subsampled_temp[::-1,:,:,:]


    if ii == 0:
        x = data_subsampled.transpose(2,0,1,3)
        y_odi = data_odi.transpose(2,0,1,3)
        y_fiso = data_fiso.transpose(2,0,1,3)
        y_ficvf = data_ficvf.transpose(2,0,1,3)
        y_gfa = data_gfa.transpose(2,0,1,3)
        y_md = data_md.transpose(2,0,1,3)
        y_ad = data_ad.transpose(2,0,1,3)
        y_fa = data_fa.transpose(2,0,1,3)
        
    else:
        x = np.concatenate((x,
                            data_subsampled.transpose(2,0,1,3)),
                           axis=0)
        y_odi = np.concatenate((y_odi,
                                data_odi.transpose(2,0,1,3)),
                               axis=0)
        y_fiso = np.concatenate((y_fiso,
                                 data_fiso.transpose(2,0,1,3)),
                                axis=0)
        y_ficvf = np.concatenate((y_ficvf,
                                  data_ficvf.transpose(2,0,1,3)),
                                 axis=0)
        y_gfa = np.concatenate((y_gfa,
                                data_gfa.transpose(2,0,1,3)),
                               axis=0)
        y_md = np.concatenate((y_md,
                               data_md.transpose(2,0,1,3)),
                              axis=0)
        y_ad = np.concatenate((y_ad,
                                data_ad.transpose(2,0,1,3)),
                               axis=0)
        y_fa = np.concatenate((y_fa,
                               data_fa.transpose(2,0,1,3)),
                              axis=0)
        
    ii += 1

print("The final data shape is: (%i, %i, %i, %i)" % x.shape)
n_samples, dim0, dim1, n_channels = x.shape

### NORMALIZATION ###
x_reshape = x.reshape(-1,n_channels)
maxs = np.amax(x_reshape, axis=0)[None, None, None, :]
x /= maxs


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


### SAVING ###
hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/max_values_2d.h5","w")
hf.create_dataset("max_values", data=maxs)
hf.close

hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/max_y_2d.h5","w")
hf.create_dataset("max_y", data=max_y)
hf.close
    
hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/x_%i_directions_2d.h5" %
               len(subsampling),"w")
hf.create_dataset("x_%i_directions" % len(subsampling), data=x)
hf.close

hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/y_odi_2d.h5","w")
hf.create_dataset("y_odi",data=y_odi)
hf.close

hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/y_fiso_2d.h5","w")
hf.create_dataset("y_fiso",data=y_fiso)
hf.close

hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/y_ficvf_2d.h5","w")
hf.create_dataset("y_ficvf",data=y_ficvf)
hf.close

hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/y_gfa_2d.h5","w")
hf.create_dataset("y_gfa",data=y_gfa)
hf.close

hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/y_md_2d.h5","w")
hf.create_dataset("y_md",data=y_md)
hf.close

hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/y_ad_2d.h5","w")
hf.create_dataset("y_ad",data=y_ad)
hf.close

hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/y_fa_2d.h5","w")
hf.create_dataset("y_fa",data=y_fa)
hf.close


plt.figure()
display.Render(x[25,:,:,:])
plt.show()
