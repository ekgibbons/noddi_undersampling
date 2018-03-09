import json
import sys

import numpy as np
from matplotlib import pyplot as plt

import nibabel as nib

class NoddiData(object):
    """
    """

    def __init__(self, patient_number=None, metadata_name=None):
        """
        """

        if metadata_name is None:
            metadata_name = "noddi_metadata.json"

        with open(metadata_name) as metadata:
            self.patient_database = json.load(metadata)

        if patient_number is None:
            print("Choose patient from the following list:\n")
            for patient_number_list in self.patient_database.keys():
                print("\t%s" % patient_number_list)

            sys.exit(0)

        self.patient_info = self.patient_database[patient_number]
                
    def get_full(self,return_nifti=False):
        """
        Returns the full (all directions) image pixel values

        Returns
        -------
        """

        return self._return_data("full", return_nifti)

    def get_odi(self,return_nifti=False):
        """
        Returns the ODI map pixel values

        Returns
        -------
        """

        return self._return_data("odi", return_nifti)

    def get_fiso(self,return_nifti=False):
        """
        Returns the FISO map image pixel values

        Returns
        -------
        """

        return self._return_data("fiso", return_nifti)


    def get_ficvf(self,return_nifti=False):
        """
        Returns the FICVF map pixel values

        Returns
        -------
        """

        return self._return_data("ficvf", return_nifti)
        

    def _return_data(self, data_type, return_nifti):

        img = nib.load(self.patient_info[data_type])
        
        if return_nifti:
            return img

        else:
            img_np = img.get_data()
            return img_np

    
        
def main():

    noddi_data = NoddiData("P100716")

    data = noddi_data.get_fiso()

    print(data.shape)
    


if __name__ == "__main__":
    main()

                  
        
