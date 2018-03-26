import glob
import ntpath
import json

def pnumber(filename):

    print(ntpath.basename(filename))
    
path_full = "/v/raid1b/khodgson/MRIdata/DTI/CNC_Imris/Stroke_patients/Stroke_DSI_Processing/Data/DeEddyed_volumes"
path_noddi = "/v/raid1b/khodgson/MRIdata/DTI/CNC_Imris/Stroke_patients/Stroke_DSI_Processing/Data/Models/NODDIMaps"

filenames_full = glob.glob("%s/*.nii.gz" % path_full)
filenames_noddi = glob.glob("%s/*.nii" % path_noddi)

patients = {}

num_patients = len(filenames_full)

ii = 0
for filename in filenames_full:
    
    filename_base = ntpath.basename(filename)
    if filename_base[7] is not "_":
        patient_number = filename_base[:8]
    else:
        patient_number = filename_base[:7]

    noddi_maps = [noddi_file for noddi_file in filenames_noddi if patient_number in noddi_file]

    patient_dict = {}
    patient_dict["full"] = filename
    try:
        patient_dict["odi"] = [odi_file for odi_file in noddi_maps if "odi." in odi_file][0]
        patient_dict["fiso"] = [fiso_file for fiso_file in noddi_maps if "fiso." in fiso_file][0]
        patient_dict["ficvf"] = [ficvf_file for ficvf_file in noddi_maps if "ficvf." in ficvf_file][0]
    except:
        print("WARNING:  noddi files missing for case %s" % patient_number)
        continue

    patients[patient_number] = patient_dict

    if (num_patients - 4) > ii:
        patient_dict["data_type"] = "train"
    else:
        patient_dict["data_type"] = "test"
    
    ii += 1

print(len(patients))

json_filename = "noddi_metadata.json"

with open(json_filename,"w") as json_filename:
    json.dump(patients,json_filename,sort_keys=True,indent=4)
