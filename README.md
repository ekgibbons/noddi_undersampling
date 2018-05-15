# Noddi processing project

This is the code for generating NODDI maps (ODI, FISO, and FICVF) rapidly and simultaneously through deep learning methods.

## Running this code

This library will require the following the dependencies:

* python3.5+
* keras2.1+
* matlab.engine
* matplotlib
* skimage0.13+
* numpy
* h5py2.7+
* scipy0.19+

If you don't have this installed, the easiest is through the anaconda virtual environments.

```
conda create -n [your_env_name] python=3.6
```

This will install the conda environment.  Now you need to install each of the pacakages either through `pip` or `conda install`.

The exception to this is the `matlab.engine` package, which will be needed separately.  This can be done by (according to this)

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

