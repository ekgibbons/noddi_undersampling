from matplotlib import pyplot as plt
import nibabel as nib 

from utils import display

path_mask = "/v/raid1b/khodgson/MRIdata/DTI/CNC_Imris/Stroke_patients/Stroke_DSI_Processing/Data/Models/Masks/P032315_MaskForNODDI.nii"

img = nib.load(path_mask)

img_np = img.get_data()

plt.figure()
display.render(img_np)
plt.show()


