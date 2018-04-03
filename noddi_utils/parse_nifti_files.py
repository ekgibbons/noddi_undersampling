import glob
import ntpath
import json

def pnumber(filename):

    print(ntpath.basename(filename))
    
path_full = ("/v/raid1b/khodgson/MRIdata/DTI/CNC_Imris/Stroke_patients/"
             "Stroke_DSI_Processing/Data/DeEddyed_volumes")
path_noddi = ("/v/raid1b/khodgson/MRIdata/DTI/CNC_Imris/Stroke_patients/"
              "Stroke_DSI_Processing/Data/Models/NODDIMaps")
path_gfa = ("/v/raid1b/khodgson/MRIdata/DTI/CNC_Imris/Stroke_patients/" 
            "Stroke_DSI_Processing/Data/Models/GFAMaps/GFA_eddy_rot_bvecs")
path_dti = ("/v/raid1b/khodgson/MRIdata/DTI/CNC_Imris/Stroke_patients/"
            "Stroke_DSI_Processing/Data/Models/ADCDTIMaps")
path_ad = ("/v/raid1b/khodgson/MRIdata/DTI/CNC_Imris/Stroke_patients/"
           "Stroke_DSI_Processing/Data/Models/ADCDTIMaps/ADmaps")


filenames_full = glob.glob("%s/*.nii.gz" % path_full)
filenames_noddi = glob.glob("%s/*.nii" % path_noddi)
filenames_gfa = glob.glob("%s/*.mat" % path_gfa)
filenames_dti = glob.glob("%s/*.nii.gz" % path_dti)
filenames_ad = glob.glob("%s/*.nii.gz" % path_ad)

patients = {}

num_patients = len(filenames_full)

ii = 0
for filename in filenames_full:
    
    filename_base = ntpath.basename(filename)
    if filename_base[7] is not "_":
        patient_number = filename_base[:8]
    else:
        patient_number = filename_base[:7]

    patient_dict = {}
    patient_dict["full"] = filename

    noddi_maps = [noddi_file for noddi_file in filenames_noddi if patient_number in noddi_file]
    try:
        patient_dict["odi"] = [odi_file for odi_file in noddi_maps if "odi." in odi_file][0]
        patient_dict["fiso"] = [fiso_file for fiso_file in noddi_maps if "fiso." in fiso_file][0]
        patient_dict["ficvf"] = [ficvf_file for ficvf_file in noddi_maps if "ficvf." in ficvf_file][0]
    except:
        print("WARNING:  noddi files missing for case %s" % patient_number)
        continue

    gfa_maps = [gfa_file for gfa_file in filenames_gfa if patient_number in gfa_file]
    try:
        patient_dict["gfa"] = [gfa_file for gfa_file in gfa_maps if "_GFA." in gfa_file][0]
    except:
        print("WARNING:  gfa files missing for case %s" % patient_number)
        continue


    # dti_maps = [dti_file for dti_file in filenames_dti if patient_number in dti_file]
    # try:
    #     patient_dict["fa"] = [fa_file for fa_file in dti_maps if "fa." in fa_file][0]
    #     patient_dict["md"] = [md_file for md_file in dti_maps if "md." in md_file][0]
    # except:
    #     print("WARNING:  dti files missing for case %s" % patient_number)
    #     continue

    # ad_maps = [ad_file for ad_file in filenames_ad if patient_number in ad_file]
    # try:
    #     patient_dict["ad"] = [ad_file for ad_file in ad_maps if "ad." in ad_file][0]
    # except:
    #     print("WARNING:  ad files missing for case %s" % patient_number)
    #     continue
    
        
    patients[patient_number] = patient_dict

    if (patient_number == "P111816") or (patient_number == "P080715"):
        patient_dict["data_type"] = "test"
    else:
        patient_dict["data_type"] = "train"
    
    ii += 1

print(len(patients))

json_filename = "noddi_metadata.json"

with open(json_filename,"w") as json_filename:
    json.dump(patients,json_filename,sort_keys=True,indent=4)
