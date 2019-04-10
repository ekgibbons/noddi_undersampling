import os
import sys
import time 

loss_type = "l1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import h5py
import numpy as np
from matplotlib import pyplot as plt

sys.path.append("/home/mirl/egibbons/noddi")

from noddi_utils import noddistudy
from noddi_utils import predict
from noddi_utils import subsampling
from recon import imtools
from utils import readhdf5
from utils import display

test_cases = ["P032315","P080715","P061114",
              "N011118A","N011118B"]


directions = [128, 64, 32, 24, 16, 8]
seeds = [100, 225, 300, 325, 400, 425, 500, 525, 600]

for patient_number in test_cases:
    for n_directions in directions:

        noddi_data = noddistudy.NoddiData(patient_number)
        
        max_y_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/max_y_2d.h5"
        max_y = readhdf5.read_hdf5(max_y_path,"max_y")
        
        data_full = noddi_data.get_full()
        
        prediction_2d = predict.model_2d(data_full, n_directions, random_seed=400)

        hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/processing/%s_%i_directions_2d.h5" %
                       (patient_number, n_directions),"w")
        hf.create_dataset("predictions", data=prediction_2d)
        hf.close

        prediction_1d_res = predict.model_1d(data_full, n_directions, random_seed=400)

        hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/processing/%s_%i_directions_1d_res.h5" %
                       (patient_number, n_directions),"w")
        hf.create_dataset("predictions", data=prediction_1d_res)
        hf.close
        
        prediction_1d = predict.golkov_multi(data_full, n_directions, random_seed=400)
        
        hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/processing/%s_%i_directions_1d.h5" %
                       (patient_number, n_directions),"w")
        hf.create_dataset("predictions", data=prediction_1d)
        hf.close

        prediction_separate_2d = predict.separate_2d(data_full, n_directions, random_seed=400)

        hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/processing/%s_%i_directions_separate_2d.h5" %
                       (patient_number, n_directions),"w")
        hf.create_dataset("predictions", data=prediction_separate_2d)
        hf.close

        prediction_separate_no_scale_2d = predict.separate_2d(data_full, n_directions,
                                                              random_seed=400, scaling=False)

        hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/processing/"
                       "%s_%i_directions_separate_no_scale_2d.h5" %
                       (patient_number, n_directions),"w")
        hf.create_dataset("predictions", data=prediction_separate_2d)
        hf.close

        
        
    # for random_seed in seeds:
    #     n_directions = 24

    #     noddi_data = noddistudy.NoddiData(patient_number)
        
    #     max_y_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/max_y_2d.h5"
    #     max_y = readhdf5.read_hdf5(max_y_path,"max_y")
        
    #     data_full = noddi_data.get_full()
        
    #     prediction_2d = predict.sampling_2d(data_full, n_directions,
    #                                      random_seed=random_seed)

    #     hf = h5py.File("/v/raid1b/egibbons/MRIdata/DTI/noddi/processing/%s_%i_directions_%i_seed_2d.h5" %
    #                    (patient_number, n_directions, random_seed),"w")
    #     hf.create_dataset("predictions", data=prediction_2d)
    #     hf.close


