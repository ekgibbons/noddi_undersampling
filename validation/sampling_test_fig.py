import sys

from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np
from scipy import stats

sys.path.append("/home/mirl/egibbons/noddi")
from noddi_utils import metrics
from noddi_utils import noddistudy
from utils import display
from utils import readhd5

test_cases = ["P032315","P080715","P020916",
              "N011118A","N011118B"]

seeds = [225, 325, 400, 425, 525, 600]



max_y_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/max_y_2d.h5"
max_y = readhd5.ReadHDF5(max_y_path,"max_y")

data = {}
data_types = ["odi", "fiso", "ficvf", "gfa"]
measurements = ["ssim", "psnr", "nrmse"]

ind = np.arange(len(data_types))
width = 0.08

for seed in seeds:
    data[seed] = {}
    for data_type in data_types:
        data[seed][data_type] = {}
    
        for measurement_type in measurements:
            data[seed][data_type][measurement_type] = ([],[])

n_directions = 24
model_type = "2d"

colors = ["blue","yellow","green","red","violet",
          "orange", "cyan", "magenta", "gray"] 

for seed in seeds:
    
    ii  = 0 
    for patient_number in test_cases:
        noddi_data = noddistudy.NoddiData(patient_number)
        
        path_data = ("/v/raid1b/egibbons/MRIdata/DTI/noddi/"
                     "processing/%s_%i_directions_%i_seed_%s.h5" %
                     (patient_number, n_directions, seed, model_type))
        prediction = readhd5.ReadHDF5(path_data,"predictions")
            
        data_odi = noddi_data.get_odi()
        data_fiso = noddi_data.get_fiso()
        data_ficvf = noddi_data.get_ficvf()
        data_gfa = noddi_data.get_gfa()

        data_odi[data_odi>max_y[0]] = max_y[0]
        data_fiso[data_fiso>max_y[1]] = max_y[1]
        data_ficvf[data_ficvf>max_y[2]] = max_y[2]
        data_gfa[data_gfa>max_y[3]] = max_y[3]


        slice_cut = 6
        if ii == 0:
            prediction_combine = prediction[:,:,slice_cut:-1-slice_cut,:]
            data_odi_combine = data_odi[:,:,slice_cut:-1-slice_cut]
            data_fiso_combine = data_fiso[:,:,slice_cut:-1-slice_cut]
            data_ficvf_combine = data_ficvf[:,:,slice_cut:-1-slice_cut]
            data_gfa_combine = data_gfa[:,:,slice_cut:-1-slice_cut]
        else:
            prediction_combine = np.concatenate((prediction_combine,
                                                 prediction[:,:,slice_cut:-1-slice_cut,:]),
                                                axis=2)
            data_odi_combine = np.concatenate((data_odi_combine,
                                               data_odi[:,:,slice_cut:-1-slice_cut]),
                                              axis=2)
            data_fiso_combine = np.concatenate((data_fiso_combine,
                                                data_fiso[:,:,slice_cut:-1-slice_cut]),
                                               axis=2)
            data_ficvf_combine = np.concatenate((data_ficvf_combine,
                                                 data_ficvf[:,:,slice_cut:-1-slice_cut]),
                                                axis=2)
            data_gfa_combine = np.concatenate((data_gfa_combine,
                                                   data_gfa[:,:,slice_cut:-1-slice_cut]),
                                              axis=2)

        ii += 1

    
    odi_all = metrics.volume_all(data_odi_combine,
                                 prediction_combine[:,:,:,0])
    fiso_all = metrics.volume_all(data_fiso_combine,
                                  prediction_combine[:,:,:,1])
    ficvf_all = metrics.volume_all(data_ficvf_combine,
                                   prediction_combine[:,:,:,2])
    gfa_all = metrics.volume_all(data_gfa_combine,
                                 prediction_combine[:,:,:,3])

    

    data[seed]["odi"] = odi_all
    data[seed]["fiso"] = fiso_all
    data[seed]["ficvf"] = ficvf_all
    data[seed]["gfa"] = gfa_all



# jj = 0
# for measurement_type in measurements:
#     for data_type in data_types:
#         _, p_value = stats.kruskal(data[100][data_type][jj],
#                                              data[225][data_type][jj],
#                                              data[300][data_type][jj],
#                                              data[400][data_type][jj],
#                                              data[500][data_type][jj],
#                                              data[600][data_type][jj])

#         print("measurement: %s, data type: %s, p-value: %f" %
#               (measurement_type, data_type, p_value))
        
#     jj += 1


    
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
linewidth_use = 1.5

x_axis_labels = ["ODI", "FISO", "FICVF", "GFA"]
y_axis_labels = ["SSIM", "PSNR", "NRMSE"]


jj = 0
for measurement_type in measurements:
    cum_width = 0
    fig, ax = plt.subplots()
    ii = 0
    for seed in seeds:
        nn = 0
        for data_type in data_types:
            data_plot = data[seed][data_type][jj]
            
            ax.boxplot(data_plot, positions=[ind[nn] + cum_width], notch=True,
                       patch_artist=True,
                       widths=abs(width),
                       boxprops=dict(facecolor='white',
                                     color=display.get_color(colors[ii]),
                                     linewidth=linewidth_use),
                       capprops=dict(color=display.get_color(colors[ii]),
                                     linewidth=linewidth_use),
                       whiskerprops=dict(color=display.get_color(colors[ii]),
                                         linewidth=linewidth_use),
                       flierprops=dict(color=display.get_color(colors[ii]),
                                       linewidth=linewidth_use,
                                       markeredgecolor=display.get_color(colors[ii])),
                       medianprops=dict(color=display.get_color(colors[ii]),
                                        linewidth=linewidth_use),
                )
            
            nn += 1

        cum_width += width + 0.015


        ii += 1

    if measurement_type == "ssim":
        ax.set_ylim((0.75, 1.0))

    ax.set_xlim([ind[0] +3*width - width/2 + 0.015*3 - 0.5,
                 ind[3] +3*width - width/2 + 0.015*3 + 0.5])
    ax.set_xticks(ind + 3*width - width/2 + 0.015*3)
    ax.set_xticklabels(x_axis_labels)
    ax.set_xlabel("parameter map")
    ax.set_ylabel(y_axis_labels[jj])
    plt.savefig("../results/fig5_seed_%s_%s.pdf"
                % (measurement_type, data_type),
                bbox_inches="tight")

    jj += 1
    
plt.show()






        # ax.bar(ind + cum_width, data_means, width, yerr=data_std,
        #        color=display.get_color(colors[ii]), capsize=2
        # )
