# Noddi processing project

This is the code for generating NODDI maps (ODI, FISO, and FICVF) rapidly and simultaneously through deep learning methods.

## Running this code

This library will require the following the dependencies:

* python3.5+
* tensorflow1.8+ (`pip install --upgrade tensorflow-gpu`)
* keras2.1+ (`pip install keras`)
* webcolors (`pip install webcolors`)
* ants (see their github for install)

* matplotlib 
* scikit-image
* numpy
* h5py
* scipy
* nibabel
* matlab.engine

If you don't have these installed, the easiest is through the anaconda virtual environments.

```
conda create -n [your_env_name] python=3.6
```

This will install the conda environment.  Now you need to install each of the pacakages either through `pip` or `conda install`.  You will need CUDA9.0 and CUDNN7.  You can get that through

```
conda install cudnn
```
The exception to this is the `matlab.engine` package, which will be needed separately.  This can be done by

```
cd "matlabroot\extern\engines\python"
python setup.py build --build-base="builddir" install --prefix="installdir"
```
To use the virtual environment, just run

```
source activate [your_env_name]
```

You will need to add my `python_utils` library to the `PYTHON_PATH` by

```
source path_setup.sh
```

## Generating the figures for the paper

To generate the figures for the paper

```
cd validation
python [whatever_figure_you_want]_fig.py
```

This should spit out the figure window pane through x11 and also save it in the `results` folder (relative to the base).  You will need to make the `results` folder first, however.  

