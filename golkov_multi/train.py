import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from keras import backend as K
from keras.callbacks import EarlyStopping
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
from utils import readhdf5

def train(n_directions):
    print("running network with %i directions" % n_directions)
    
    n_gpu = 1
    n_epochs = 100
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

    print("Loading data...")
    x = readhdf5.read_hdf5(x_path,"x_%i_directions" % n_directions)
    y_odi = readhdf5.read_hdf5(y_odi_path,"y_odi")
    y_fiso = readhdf5.read_hdf5(y_fiso_path,"y_fiso")
    y_ficvf = readhdf5.read_hdf5(y_ficvf_path,"y_ficvf")
    y_gfa = readhdf5.read_hdf5(y_gfa_path,"y_gfa")
    print("Data is loaded...")
    
    n_samples, _ = x.shape

    print(y_odi.shape)
    print(y_fiso.shape)
    print(y_ficvf.shape)
    print(y_gfa.shape)
    
    y = np.concatenate((y_odi, y_fiso, y_ficvf, y_gfa),
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

    stopping = EarlyStopping(monitor="val_loss", min_delta=0,
                             patience=30, verbose=0, mode='auto')
    
    model.fit(x=x,
              y=y,
              batch_size=batch_size_multi_gpu,
              epochs=n_epochs,
              verbose=2,
              callbacks=[checkpointer, reduce_lr, stopping],
              validation_split=0.2,
              shuffle=True,
    )
    
    print("trained %i direction model" % n_directions)

    
def main(argv):

    if (len(argv) == 1) or (argv[1] == "64"):
        n_directions = 64
    else:
        n_directions = int(argv[1])
    
    train(n_directions)

    
if __name__ == "__main__":

    main(sys.argv)
    
