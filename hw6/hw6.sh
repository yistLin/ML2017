#!/bin/bash
# hw6.sh
# python version: python3.6

if [ $# != 2 ]; then
    echo "usage: ./hw6.sh <test data path> <output path>"
    exit 1
fi

TEST_DATA_PATH=$1
OUTPUT_PATH=$2

echo "TEST_DATA_PATH = $TEST_DATA_PATH"
echo "OUTPUT_PATH    = $OUTPUT_PATH"

python3 mf_predict.py \
    $TEST_DATA_PATH \
    --model_path models/mf_nobias.hdf5 \
    --output_path $OUTPUT_PATH \

