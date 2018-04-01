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
                
        self.patient_info = self.patient_database[patient_number]

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
            return img_np

        
def main():
    from matplotlib import pyplot as plt
    
    import noddistudy
    from utils import display
    
    noddi_data = noddistudy.NoddiData("P100716")

    data = noddi_data.get_adc()
    print(np.max(data[:,:,13]))

    data[data > 5e-8] = 5e-8
    
    plt.figure()
    display.Render(data)

    plt.figure()
    plt.imshow(abs(data[:,:,-1-3]))
    plt.show()

if __name__ == "__main__":
    main()

                  
        
