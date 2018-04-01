import glob
import sys

from matplotlib import pyplot as plt
import numpy as np

sys.path.append("/home/mirl/egibbons/noddi")

from noddi_utils import noddistudy
from utils import display


def gensamples(num_samples, patient_number="N011618",
               shuffle=False, verbose=False, plot=False):
    """Sampling pattern generator for the directions in DSI study.

    Parameters
    ----------
    num_samples: int
        Number of directions to be sampled
    patient_number: str
        String of the patient number used in the path
    shuffle: Bool
        Whether or not to suffle the sampling list
    verbose: Bool
        If the sampling is printed out when called
    plot: Bool
        If a normalized test example is printed
    

    Returns
    -------
    subsampling: array_like
        1D array of the sampling indices
    """


    
    path_to_data = ("/v/raid1b/khodgson/MRIdata/DTI/CNC_Imris/"
                    "Stroke_patients/Stroke_DSI_Processing/Data/Bvals")
    
    patient_number = "N011618"
    filename = "%s/%s_bvals.txt" % (path_to_data, patient_number)
    
    with open (filename) as bvalue_file:
        line = []
        for bvalue in bvalue_file.readlines()[0].split():
            line.append(int(float(bvalue)))

    bvalues = np.array(line)
    
    # num_bins = 30
    
    # plt.figure()
    # plt.hist(bvalues, num_bins,alpha=0.5)
    
    # now divide the directions into shells
    shell1_divide = 1000
    shell2_divide = 2000
    
    bvalue_dict = {}
    bvalue_dict["b0"] = []
    bvalue_dict["shell1"] = []
    bvalue_dict["shell2"] = []
    bvalue_dict["shell3"] = []

    for ii in range(bvalues.shape[0]):
        if bvalues[ii] == 0:
            bvalue_dict["b0"].append(ii)
        elif (0 < bvalues[ii]) and (bvalues[ii] <= shell1_divide):
            bvalue_dict["shell1"].append(ii)
        elif (shell1_divide < bvalues[ii]) and (bvalues[ii] <= shell2_divide):
            bvalue_dict["shell2"].append(ii)
        elif (shell2_divide < bvalues[ii]):
            bvalue_dict["shell3"].append(ii)

    num_b0 = round(num_samples*len(bvalue_dict["b0"])/bvalues.shape[0])
    num_shell1 = round(num_samples*len(bvalue_dict["shell1"])/bvalues.shape[0])
    num_shell2 = round(num_samples*len(bvalue_dict["shell2"])/bvalues.shape[0])
    num_shell3 = num_samples - (num_b0 + num_shell1 + num_shell2)

    np.random.seed(seed=100)
    b0 = np.random.choice(bvalue_dict["b0"], num_b0, replace=False)
    shell1 = np.random.choice(bvalue_dict["shell1"], num_shell1, replace=False)
    shell2 = np.random.choice(bvalue_dict["shell2"], num_shell2, replace=False)
    shell3 = np.random.choice(bvalue_dict["shell3"], num_shell3, replace=False)
    
    subsampling = np.concatenate((b0,shell1,shell2,shell3),
                                 axis=0)


    if shuffle is True:
        np.random.shuffle(subsampling)
    
    if verbose is True:
        print("b0 sampling:")
        print(b0)
        print("shell1 sampling:")
        print(shell1)
        print("shell2 sampling:")
        print(shell2)
        print("shell3 sampling:")
        print(shell3)
        
        print("num_samples")
        print(subsampling)

    if plot is True:
        study = noddistudy.NoddiData(patient_number)
        data_full = study.get_full()
        
        data = data_full[:,:,:,subsampling]
        
        maxs = np.amax(data.reshape(-1,num_samples),
                       axis=0).squeeze()
        
        print(maxs)
        
        data /= maxs[None,None,None,:]
        
        plt.figure()
        display.Render(data[:,:,25,:])
        plt.show()

    return subsampling
