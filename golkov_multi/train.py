import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from keras.utils.training_utils import multi_gpu_model
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

import model1d
from utils import display
from utils import readhd5

if (len(sys.argv) == 1) or (sys.argv[1] == "64"):                            
    n_directions = 64                                                        
else:                                                                        
    n_directions = int(sys.argv[1])                                          

print("running network with %i directions" % n_directions)
    
n_gpu = 1
n_epochs = 200
batch_size = 10000
learning_rate = 1e-3

image_size = (n_directions,)

model = model1d.fc_1d(image_size)
    
optimizer = Adam(lr=learning_rate)

model.compile(optimizer=optimizer,
              loss="mean_squared_error",
              )

x_path = ("/v/raid1b/egibbons/MRIdata/DTI/noddi/x_%i_directions_1d.h5" 
          % n_directions)
y_odi_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_odi_1d.h5"
y_fiso_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_fiso_1d.h5"
y_ficvf_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_ficvf_1d.h5"
y_gfa_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_gfa_1d.h5"
# y_md_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_md_1d.h5"
# y_ad_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_ad_1d.h5"
# y_fa_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_fa_1d.h5"

print("Loading data...")
x = readhd5.ReadHDF5(x_path,"x_%i_directions" % n_directions)
y_odi = readhd5.ReadHDF5(y_odi_path,"y_odi")
y_fiso = readhd5.ReadHDF5(y_fiso_path,"y_fiso")
y_ficvf = readhd5.ReadHDF5(y_ficvf_path,"y_ficvf")
y_gfa = readhd5.ReadHDF5(y_gfa_path,"y_gfa")
# y_md = readhd5.ReadHDF5(y_md_path,"y_md")
# y_ad = readhd5.ReadHDF5(y_ad_path,"y_ad")
# y_fa = readhd5.ReadHDF5(y_fa_path,"y_fa")
print("Data is loaded...")

n_samples, _ = x.shape

y = np.concatenate((y_odi, y_fiso, y_ficvf, y_gfa),
                    # y_md, y_ad, y_fa),
                   axis=1)

batch_size_multi_gpu = n_gpu*batch_size

tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=1,
                          batch_size=batch_size_multi_gpu,
                          write_graph=True,
                          write_grads=True,
                          write_images=True,
                          embeddings_freq=0,
                          embeddings_layer_names=None,
                          embeddings_metadata=None)

save_path = ("/v/raid1b/egibbons/models/noddi-%i_golkov_multi.h5" %
             n_directions)
checkpointer = ModelCheckpoint(filepath=save_path,
                               verbose=1, monitor="val_loss", save_best_only=True,
                               save_weights_only=True, period=25)

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1,
                              patience=5, min_lr=1e-7) 

model.fit(x=x,
          y=y,
          batch_size=batch_size_multi_gpu,
          epochs=n_epochs,
          verbose=1,
          callbacks=[checkpointer, reduce_lr],
          validation_split=0.2,
          shuffle=True,
          )

