#!/bin/bash

echo "Appending PYTHONPATH"
PYTHONPATH=$PYTHONPATH:/home/mirl/egibbons/python_utils

export PYTHONPATH

echo ${PYTHONPATH}
