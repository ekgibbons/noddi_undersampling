import sys

from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np

sys.path.append("/home/mirl/egibbons/noddi")
from noddi_utils import metrics
from noddi_utils import noddistudy
from utils import display
from utils import readhdf5

test_cases = ["P032315","P061815","P020916","N011118A",
              "N011118B","P072216","P082616"]

directions = [128, 64, 32, 24, 16]

max_y_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/max_y_2d.h5"
max_y = readhdf5.read_hdf5(max_y_path,"max_y")

data = {}
models = ["2d", "separate_2d"]
data_types = ["odi", "fiso", "ficvf", "gfa"]
measurements = ["SSIM", "PSNR", "NRMSE"]


for model_type in models:
    data[model_type] = {}
    
    for data_type in data_types:
        data[model_type][data_type] = {}
        
        for measurement_type in measurements:
            data[model_type][data_type][measurement_type] = ([],[])


for model_type in models:
    for n_directions in directions:            
        ii  = 0 
        for patient_number in test_cases:
            noddi_data = noddistudy.NoddiData(patient_number)

            print("%i directions" % n_directions)
            if model_type == "2d":
                print("2D model")
            else:
                print("1d model")

            path_data = ("/v/raid1b/egibbons/MRIdata/DTI/noddi/"
                         "processing/%s_%i_directions_%s.h5" %
                         (patient_number, n_directions,model_type))
            prediction = readhdf5.read_hdf5(path_data,"predictions")
            
            data_odi = noddi_data.get_odi()
            data_fiso = noddi_data.get_fiso()
            data_ficvf = noddi_data.get_ficvf()
            data_gfa = noddi_data.get_gfa()

            data_odi[data_odi>max_y[0]] = max_y[0]
            data_fiso[data_fiso>max_y[1]] = max_y[1]
            data_ficvf[data_ficvf>max_y[2]] = max_y[2]
            data_gfa[data_gfa>max_y[3]] = max_y[3]
            
            if ii == 0:
                prediction_combine = prediction[:,:,5:-1-5,:]
                data_odi_combine = data_odi[:,:,5:-1-5]
                data_fiso_combine = data_fiso[:,:,5:-1-5]
                data_ficvf_combine = data_ficvf[:,:,5:-1-5]
                data_gfa_combine = data_gfa[:,:,5:-1-5]
            else:
                prediction_combine = np.concatenate((prediction_combine,
                                                     prediction[:,:,5:-1-5,:]),
                                                    axis=2)
                data_odi_combine = np.concatenate((data_odi_combine,
                                                   data_odi[:,:,5:-1-5]),
                                                  axis=2)
                data_fiso_combine = np.concatenate((data_fiso_combine,
                                                    data_fiso[:,:,5:-1-5]),
                                                   axis=2)
                data_ficvf_combine = np.concatenate((data_ficvf_combine,
                                                     data_ficvf[:,:,5:-1-5]),
                                                    axis=2)
                data_gfa_combine = np.concatenate((data_gfa_combine,
                                                   data_gfa[:,:,5:-1-5]),
                                                  axis=2)

            ii += 1
            
        odi_all = metrics.volume_all(prediction_combine[:,:,:,0],
                                     data_odi_combine)
        fiso_all = metrics.volume_all(prediction_combine[:,:,:,1],
                                      data_fiso_combine)
        ficvf_all = metrics.volume_all(prediction_combine[:,:,:,2],
                                       data_ficvf_combine)
        gfa_all = metrics.volume_all(prediction_combine[:,:,:,3],
                                     data_gfa_combine)
        
        print("\t--------ODI--------")
        print("\tSSIM: %f +/- %f" % odi_all[0])
        print("\tPSNR: %f +/- %f" % odi_all[1])
        print("\tMSE: %f +/- %f" % odi_all[2])
        
        print("\t--------FISO--------")
        print("\tSSIM: %f +/- %f" % fiso_all[0])
        print("\tPSNR: %f +/- %f" % fiso_all[1])
        print("\tMSE: %f +/- %f" % fiso_all[2])
        
        print("\t--------FICVF--------")
        print("\tSSIM: %f +/- %f" % ficvf_all[0])
        print("\tPSNR: %f +/- %f" % ficvf_all[1])
        print("\tMSE: %f +/- %f" % ficvf_all[2])
        
        print("\t--------GFA--------")
        print("\tSSIM: %f +/- %f" % gfa_all[0])
        print("\tPSNR: %f +/- %f" % gfa_all[1])
        print("\tMSE: %f +/- %f" % gfa_all[2])

        jj = 0
        for measurement_type in measurements:
            for ii in range(2):
                data[model_type]["odi"][measurement_type][ii].append(odi_all[jj][ii])
                data[model_type]["fiso"][measurement_type][ii].append(fiso_all[jj][ii])
                data[model_type]["ficvf"][measurement_type][ii].append(ficvf_all[jj][ii])
                data[model_type]["gfa"][measurement_type][ii].append(gfa_all[jj][ii])
            jj += 1

rcParams['font.sans-serif'] = ['Verdana']

params = {'lines.linewidth': 1.5,
          'axes.linewidth': 1.5,
          'axes.labelsize': 24,
          'font.size': 16,
          'lines.markeredgewidth':1.5,
          'ytick.labelsize': 18,
          'ytick.major.size': 8,
          'ytick.major.width': 1.5,
          'xtick.labelsize': 18,
          'xtick.major.size': 8,
          'xtick.major.width': 1.5}

rcParams.update(params)

for data_type in data_types:
    for measurement_type in measurements:
        plt.figure()
        for model_type in models:
            if model_type == "2d":
                color_use = "steelblue"
            else:
                color_use = "goldenrod"

                
            plt.errorbar(directions,
                         data[model_type][data_type][measurement_type][0],
                         yerr=data[model_type][data_type][measurement_type][1],
                         label=model_type, fmt="-o", capsize=5, elinewidth=2,
                         markeredgewidth=2, c=color_use)

        plt.legend()
        plt.xlabel("Number of directions sampled")
        plt.ylabel(measurement_type)
        plt.title(data_type)
        plt.savefig("%s_%s.pdf" % (measurement_type, data_type),
                    bbox_inches="tight")
plt.show()


        
        # if model_type == "2d":
        #     prediction_2d = prediction
        # else:
        #     prediction_1d = prediction


    # for ii in range(4):
    #     slice_use = 25*ii
    #     montage_1 = np.concatenate((data_odi[:,:,slice_use],
    #                                 data_fiso[:,:,slice_use],
    #                                 data_ficvf[:,:,slice_use],
    #                                 data_gfa[:,:,slice_use]/0.5),
    #                                axis=1)
        
    #     montage_2 = np.concatenate((prediction_1d[:,:,slice_use,0],
    #                                 prediction_1d[:,:,slice_use,1],
    #                                 prediction_1d[:,:,slice_use,2],
    #                                 prediction_1d[:,:,slice_use,3]/0.5),
    #                                axis=1)
        
    #     montage_3 = np.concatenate((prediction_2d[:,:,slice_use,0],
    #                                 prediction_2d[:,:,slice_use,1],
    #                                 prediction_2d[:,:,slice_use,2],
    #                                 prediction_2d[:,:,slice_use,3]/0.5),
    #                                axis=1)
        
    #     montage_combine = np.concatenate((montage_1,
    #                                       montage_2,
    #                                       abs(montage_1 - montage_2),
    #                                       montage_3,
    #                                       abs(montage_1 - montage_3)),
    #                                      axis=0)
        
    #     plt.figure()
    #     plt.imshow(montage_combine)
    #     plt.title("ODI, FISO, FICVF, GFA")
    #     plt.axis("off")
    #     plt.show()
        
