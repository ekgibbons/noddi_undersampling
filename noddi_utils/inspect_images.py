from matplotlib import pyplot as plt
import json
import os
import sys
import time

from utils import display
from utils import readhd5


sys.path.append("/home/mirl/egibbons/noddi")

from noddi_utils import noddistudy

metadata_name = "/home/mirl/egibbons/noddi/noddi_utils/noddi_metadata.json"

with open(metadata_name) as metadata:
    database = json.load(metadata)

print(database.keys())

for patient_name in database.keys():
    study = noddistudy.NoddiData(patient_name)
    data = study.get_full()
    
    plt.figure()
    display.Render(data[:,:,25,:])
    plt.show()

# list_names = noddistudy.NoddiData()
