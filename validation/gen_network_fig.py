import time

from matplotlib import pyplot as plt
import numpy as np

from utils import readhdf5

n_directions = 32

x_path = ("/v/raid1b/egibbons/MRIdata/DTI/noddi/x_%i_directions_2d.h5" %
          n_directions)
y_odi_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_odi_2d.h5"
y_fiso_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_fiso_2d.h5"
y_ficvf_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_ficvf_2d.h5"
y_gfa_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_gfa_2d.h5"

print("Loading data...")

start = time.time()
y_odi = readhdf5.read_hdf5(y_odi_path,"y_odi")
y_fiso = readhdf5.read_hdf5(y_fiso_path,"y_fiso")
y_ficvf = readhdf5.read_hdf5(y_ficvf_path,"y_ficvf")
y_gfa = readhdf5.read_hdf5(y_gfa_path,"y_gfa")

x = readhdf5.read_hdf5(x_path,"x_%i_directions" % n_directions).transpose(0,2,1,3)[:,::-1,::-1,:]
y = np.concatenate((y_odi, y_fiso, y_ficvf, y_gfa),
                   axis=3).transpose(0,2,1,3)[:,::-1,::-1,:]


for ii in range(x.shape[3]):
    plt.figure()
    plt.imshow(x[25,:,:,ii].squeeze(),cmap="gray")
    plt.axis("off")
    plt.axis("equal")
    plt.savefig("../results/x_%i.pdf" % ii, bbox_inches="tight")

for ii in range(y.shape[3]):
    plt.figure()
    if 0:
        plt.imshow(y[25,:,:,ii].squeeze(),cmap="hot")
    else:
        plt.imshow(y[25,:,:,ii].squeeze(),cmap="gray")
        
    plt.axis("off")
    plt.axis("equal")
    plt.savefig("../results/y_%i.pdf" % ii, bbox_inches="tight")

# plt.show()
