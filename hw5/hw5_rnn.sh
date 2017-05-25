#!/bin/bash

# python version: python3.6

TEST_DATA_PATH=$1
OUTPUT_PATH=$2

echo "TEST_DATA_PATH = $TEST_DATA_PATH"
echo "OUTPUT_PATH    = $OUTPUT_PATH"

echo "[PREDICT] generate $OUT1 from pre-trained model."
python3 rnn.py \
    --test_data $TEST_DATA_PATH \
    --class_path models/rnn-classifier.pkl \
    --model_path models/rnn-model-val_0.5141.hdf5 \
    --output_path $OUTPUT_PATH \
    --predict

