import sys

from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np
from scipy import stats
from statsmodels.stats import multitest

sys.path.append("/home/mirl/egibbons/noddi")
from noddi_utils import metrics
from noddi_utils import noddistudy
from utils import display
from utils import readhdf5

test_cases = ["P032315","P061815","P020916","N011118A",
              "N011118B","P072216","P082616"]

directions = [128, 64, 32, 24, 16, 8]

max_y_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/max_y_2d.h5"
max_y = readhdf5.read_hdf5(max_y_path,"max_y")

data = {}
models = ["2d", "separate_no_scale_2d"]
data_types = ["odi", "fiso", "ficvf", "gfa"]
measurements = ["SSIM", "PSNR", "NRMSE"]

num_slices = 7*48

for model_type in models:
    data[model_type] = {}
    
    for data_type in data_types:
        data[model_type][data_type] = {}
        
for model_type in models:
    for n_directions in directions:            
        ii  = 0 
        for patient_number in test_cases:
            noddi_data = noddistudy.NoddiData(patient_number)

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

        for measurement_type in measurements:
            data[model_type]["odi"][n_directions] = odi_all
            data[model_type]["fiso"][n_directions] = fiso_all
            data[model_type]["ficvf"][n_directions] = ficvf_all
            data[model_type]["gfa"][n_directions] = gfa_all

p_values = {}
for measurement_type in measurements:
    p_values[measurement_type] = []
        
for data_type in data_types:
    print(data_type)
    ii = 0
    for measurement_type in measurements:
        print(measurement_type)
        for n_directions in directions:

            _, p_value_wilcoxon = stats.wilcoxon(data["separate_no_scale_2d"][data_type][n_directions][ii],
                                                     data["2d"][data_type][n_directions][ii])
                        
            p_values[measurement_type].append(p_value_wilcoxon)
            
            print("\tdirections %i, p-value wilcoxon: %f" %
                  (n_directions, p_value_wilcoxon))
        ii += 1

for measurement_type in measurements:
    reject, p_values_correct, _, _ = multitest.multipletests(
        p_values[measurement_type],
        alpha=0.005,
        method="holm"
    )

    print("measurement type: %s" % measurement_type)
    for p_value_corrected in p_values_correct:
        print("value: %.6f" % p_value_corrected)

        
            
rcParams['font.sans-serif'] = ['Verdana']

params = {'axes.titlesize' : 35,
          'lines.linewidth': 1.5,
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

data_type_display = ["ODI", "FISO", "FICVF", "GFA"]

fig, axs = plt.subplots(2,2,figsize=(16,8))
axs = axs.ravel()

mm = 0
for data_type in data_types:

    ii = 0
    for measurement_type in measurements:
        if ii != 0:
            continue

        index_use = mm
        
        for model_type in models:
            
            absolute_width = 2.
            if model_type == "2d":
                color_use = display.get_color("red") # "steelblue"
                width = -absolute_width
            else:
                color_use = display.get_color("blue") # "goldenrod"
                width = absolute_width

            linewidth_use = 1.5
                
            nn = 0
            positions = [10 + kk*10  for kk in range(6)]

            for jj, data_use in sorted(data[model_type][data_type].items()):
                
                data_plot = data_use[ii]
                position = positions[nn] + width*0.6
            
                axs[index_use].boxplot(data_plot, positions=[position], notch=True,
                                       patch_artist=True,
                                       widths=abs(width),
                                       boxprops=dict(facecolor='white',
                                                     color=color_use,
                                                     linewidth=linewidth_use),
                                       capprops=dict(color=color_use,
                                                     linewidth=linewidth_use),
                                       whiskerprops=dict(color=color_use,
                                                         linewidth=linewidth_use),
                                       flierprops=dict(color=color_use,
                                                       linewidth=linewidth_use,
                                                       markeredgecolor=color_use),
                                       medianprops=dict(color=color_use,
                                             linewidth=linewidth_use),
                )

                nn += 1

        axs[index_use].set_xlim([positions[0] - 5, positions[5] + 5])
        axs[index_use].set_xticks(positions)
        axs[index_use].set_xticklabels(["8", "16", "24", "32", "64", "128"])
        axs[index_use].set_title(data_type_display[mm])
        
        ii += 1
        mm += 1

plt.tight_layout(pad=0.5, w_pad=2.0, h_pad=1.0)
plt.savefig("../results/fig6_metric_total.pdf",
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
        
