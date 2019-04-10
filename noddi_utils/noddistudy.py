import json
import os
import sys

from matplotlib import pyplot as plt
import nibabel as nib
import numpy as np

from utils import matutils

class NoddiData(object):
    """This class initiates an object that will read in the study 
    information file and returns the appropriate data in a clean 
    fashion.

    Arguments
    ---------
    patient_number: str (optional)
        The number of the study in the DiBella format ("P------").
        If nothing is passed then it will list all of the patient
        numbers to choose from.
    metadata_name:  str (optional)
        The name of the .json file to read the data from.  If nothing
        is passed then it will default the file already in the
        folder.
    """

    def __init__(self, patient_number=None, metadata_name=None):
        """
        """

        if metadata_name is None:
            dir_name = os.path.dirname(__file__)
            metadata_name = "%s/noddi_metadata.json" % (dir_name)

        with open(metadata_name) as metadata:
            self.patient_database = json.load(metadata)

        if patient_number is None:
            print("Choose patient from the following list:\n")
            for patient_number_list in self.patient_database.keys():
                print("\t%s" % patient_number_list)
            sys.exit()

        if patient_number is "test":
            self.list_test_data = []
            for patient_number_list in self.patient_database.keys():
                data_type = self.patient_database[patient_number_list]["data_type"]
                if data_type == "test":
                    self.list_test_data.append(patient_number_list)
        else:        
            self.patient_info = self.patient_database[patient_number]
        
    def get_test(self):
        """
        Returns the test data
    
        Returns
        -------
        data_type: dict of arrays
            The test data in a dictionary for each patient for all
            metrics and full data
        """

        data_test_dict = {}
        for patient_number in self.list_test_data:
            self.patient_info = self.patient_database[patient_number]
            data_patient = {}
            data_patient["odi"] = self._return_data("odi",False)
            data_patient["fiso"] = self._return_data("fiso",False)
            data_patient["ficvf"] = self._return_data("ficvf",False)
            data_patient["gfa"] = matutils.MatReader(self.patient_info["gfa"],
                                                     keyName="GFA")
            data_patient["full"] = self._return_data("full",False)
            data_test_dict[patient_number] = data_patient

        return data_test_dict

        
    def get_type(self):
        """
        Returns the type of data (training versus testing)
    
        Returns
        -------
        data_type:  str
            The type of data ("train" or "test")
        """

        return self.patient_info["data_type"]
        
        
    def get_full(self,return_nifti=False):
        """
        Returns the full (all directions) image pixel values

        Arguments
        ---------
        return_nifti: bool
            Whether or not you want the nibabel object or numpy
            array on return
        
        Returns
        -------
        data:  array_like
            The full 4D dataset (x, y, d, z)
        """

        return self._return_data("full", return_nifti)

    def get_raw(self,return_nifti=False):
        """
        Returns the raw (all directions) image pixel values

        Arguments
        ---------
        return_nifti: bool
            Whether or not you want the nibabel object or numpy
            array on return
        
        Returns
        -------
        data:  array_like
            The full 4D dataset (x, y, d, z)
        """

        return self._return_data("raw", return_nifti)
 
    def get_odi(self,return_nifti=False):
        """
        Returns the ODI map pixel values

        Arguments
        ---------
        return_nifti: bool
            Whether or not you want the nibabel object or numpy
            array on return

        Returns
        -------
        data:  array_like
            The 3D ODI map (x, y, z)
        """

        return self._return_data("odi", return_nifti)

    def get_fiso(self,return_nifti=False):
        """
        Returns the FISO map image pixel values

        Arguments
        ---------
        return_nifti: bool
            Whether or not you want the nibabel object or numpy
            array on return

        Returns
        -------
        data:  array_like
            The 3D FISO map (x, y, z)
        """

        return self._return_data("fiso", return_nifti)


    def get_ficvf(self,return_nifti=False):
        """
        Returns the FICVF map pixel values

        Arguments
        ---------
        return_nifti: bool
            Whether or not you want the nibabel object or numpy
            array on return

        Returns
        -------
        data:  array_like
            The 3D FICVF map (x, y, z)
        """

        return self._return_data("ficvf", return_nifti)

    
    def get_gfa(self,return_nifti=False):
        """
        Returns the FICVF map pixel values

        Arguments
        ---------
        return_nifti: bool
            Whether or not you want the nibabel object or numpy
            array on return

        Returns
        -------
        data:  array_like
            The 3D gfa map (x, y, z)
        """

        return matutils.MatReader(self.patient_info["gfa"],
                                  keyName="GFA")


    def get_fa(self,return_nifti=False):
        """
        Returns the fractional anistropy map pixel values

        Arguments
        ---------
        return_nifti: bool
            Whether or not you want the nibabel object or numpy
            array on return

        Returns
        -------
        data:  array_like
            The 3D FA map (x, y, z)
        """

        return self._return_data("fa", return_nifti)

    
    def get_md(self,return_nifti=False):
        """
        Returns the mean diffusivity map pixel values

        Arguments
        ---------
        return_nifti: bool
            Whether or not you want the nibabel object or numpy
            array on return

        Returns
        -------
        data:  array_like
            The 3D MD map (x, y, z)
        """

        md = self._return_data("md", return_nifti)
        md[md > 1e-8] = 1e-8
        
        return 1e6*md


    def get_ad(self,return_nifti=False):
        """
        Returns the AD map pixel values

        Arguments
        ---------
        return_nifti: bool
            Whether or not you want the nibabel object or numpy
            array on return

        Returns
        -------
        data:  array_like
            The 3D AD map (x, y, z)
        """

        ad = self._return_data("ad", return_nifti)
        ad[ad > 1e-8] = 1e-8
        
        return 1e6*ad
    

    def _return_data(self, data_type, return_nifti):

        img = nib.load(self.patient_info[data_type])
        
        if return_nifti:
            return img

        else:
            img_np = img.get_data()

            mask_condition = (self.patient_info["mask"] is not None and
                              (data_type == "odi" or data_type == "fiso" or
                               data_type == "ficvf" or data_type == "gfa"))
            if mask_condition:
                mask = nib.load(self.patient_info["mask"])
                img_np *= mask.get_data()
            
            return img_np

        
def main():
    from matplotlib import pyplot as plt
    
    import noddistudy
    from utils import display
    
    noddi_data = noddistudy.NoddiData("test")
    test_data = noddi_data.get_test()
    print(test_data.keys())
    print(test_data["N011118A"].keys())


if __name__ == "__main__":
    main()

                  
        
