import sys
import time

import matlab.engine
from matplotlib import markers
from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np

sys.path.append("/home/mirl/egibbons/noddi")
from utils import display

start = time.time()
eng = matlab.engine.start_matlab()
print("\nTime to start engine: %f" % (time.time() - start))

eng.addpath(eng.genpath("/home/mirl/egibbons/noddi/validation/fugl_meyer"))
eng.addpath(eng.genpath("/v/raid1b/khodgson/MRIdata/DTI/CNC_Imris/Stroke_patients/"
                        "Stroke_DSI_Processing/Scripts/Processing_for_Eric"))
eng.addpath(eng.genpath("/v/raid1b/gadluru/softs/freesurfer/matlab"))
            
directions = [128, 64, 32, 24, 16, 8]
patients = ["P032315","P080715","P061114"]

fm_scores = {}
for patient_number in patients:
    print(patient_number)
    fm_scores[patient_number] = {}
    for n_directions in directions:
        fm = eng.predict_fm_score(patient_number, n_directions, nargout=3)
        print(fm)
        fm_scores[patient_number][n_directions] = fm

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

        
positions = [10 + kk*10  for kk in range(8)]
colors = ["red", "blue", "green"]


spacing = 0.2
width = 2
start_delay = -(width + spacing)
          
fig, ax = plt.subplots()

for ii in range(2):
    cum_width = start_delay
    for jj, patient_number in enumerate(sorted(patients)):
        ax.bar(positions[ii] + cum_width,
               fm_scores[patient_number][n_directions][ii+1],
               width,
               color=display.get_color(colors[jj])
        )

        cum_width += width + spacing

for ii, n_directions in enumerate(sorted(directions)):
    cum_width = start_delay
    
    for jj, patient_number in enumerate(sorted(patients)):
        ax.bar(positions[ii + 2] + cum_width,
               fm_scores[patient_number][n_directions][0],
               width,
               color=display.get_color(colors[jj])
            )

        cum_width += width + spacing

ax.set_xticks(positions)
ax.set_xticklabels(["ref.", "full", "8", "16", "24", "32", "64", "128"])
ax.set_ylabel("Fugl-Meyer scores")
ax.set_xlabel("number of directions")
plt.savefig("../../results/fig7_fm_scores.pdf",
            bbox_inches="tight")

plt.figure()
for ii, n_directions in enumerate(sorted(directions)):
    for patient_number in patients:

        x_position = positions[ii+1]
        y_undersampled = (fm_scores[patient_number][n_directions][0] -
                          fm_scores[patient_number][n_directions][2])
        plt.scatter(x_position, y_undersampled, facecolors="none",
                        edgecolors=display.get_color("blue"), s=80)

        if ii == 5:
            y_full = (fm_scores[patient_number][n_directions][1] -
                      fm_scores[patient_number][n_directions][2])
            plt.scatter(positions[0], y_full, facecolors="none",
                        edgecolors=display.get_color("red"), s=80)

    
plt.plot(np.linspace(positions[0]-5, positions[6]+5, 30),
         np.zeros(30),"k--")
            
plt.xticks(positions,["full", "8", "16", "24", "32", "64", "128"])
plt.xlim([positions[0] - 5, positions[6] + 5])
plt.ylim([-30, 30])
plt.ylabel("Fugl-Meyer errors")
plt.xlabel("number of directions")
plt.savefig("../../results/fig7_fm_scores_diffs.pdf",
            bbox_inches="tight")

plt.show()
