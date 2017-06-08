#!/bin/bash
# hw6.sh
# python version: python3.6

if [ $# != 2 ]; then
    echo "usage: ./hw6.sh <data dir> <output path>"
    exit 1
fi

DATA_DIR=$1
OUTPUT_PATH=$2

echo "DATA_DIR    = $DATA_DIR"
echo "OUTPUT_PATH = $OUTPUT_PATH"

python3 mf_predict.py \
    "$DATA_DIR/test.csv" \
    --model_path models/mf_nobias.hdf5 \
    --output_path $OUTPUT_PATH \

