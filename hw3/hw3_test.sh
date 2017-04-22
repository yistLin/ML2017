#!/bin/bash

# python version: python3.5
# python3 predict.py $1 model-0.7073.hdf5 $2
python3 ensemble_predict.py $1 $2 model-0.6983.hdf5 model-0.6990.hdf5 model-0.7047.hdf5 model-0.7057.hdf5 model-0.7073.hdf5
