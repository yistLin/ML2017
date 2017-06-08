#!/bin/bash
# hw6_best.sh
# python version: python3.6

if [ $# != 2 ]; then
    echo "usage: ./hw6_best.sh <data dir> <output path>"
    exit 1
fi

DATA_DIR=$1
OUTPUT_PATH=$2

echo "DATA_DIR = $DATA_DIR"
echo "OUTPUT_PATH    = $OUTPUT_PATH"

python3 mf_bias.py \
    "$DATA_DIR/test.csv" \
    --model_path models/mf_bias.hdf5 \
    --output_path $OUTPUT_PATH \
    --predict

