#!/bin/bash

# python version: python3.6

TEST_DATA_PATH=$1
OUTPUT_PATH=$2

echo "TEST_DATA_PATH = $TEST_DATA_PATH"
echo "OUTPUT_PATH    = $OUTPUT_PATH"

OUT1="temp_rnn_output_1.csv"
OUT2="temp_rnn_output_2.csv"
OUT3="temp_rnn_output_3.csv"
OUT4="temp_svm_output_1.csv"
OUT5="temp_svm_output_2.csv"

echo "[PREDICT] generate $OUT1 from pre-trained model."
python3 rnn.py \
    --test_data $TEST_DATA_PATH \
    --class_path models/rnn-classifier.pkl \
    --model_path models/rnn-model-val_0.5141.hdf5 \
    --output_path $OUT1 \
    --predict

echo "[PREDICT] generate $OUT2 from pre-trained model."
python3 rnn.py \
    --test_data $TEST_DATA_PATH \
    --class_path models/rnn-classifier.pkl \
    --model_path models/rnn-model-val_0.4889.hdf5 \
    --output_path $OUT2 \
    --predict

echo "[PREDICT] generate $OUT3 from pre-trained model."
python3 rnn.py \
    --test_data $TEST_DATA_PATH \
    --class_path models/rnn-classifier.pkl \
    --model_path models/rnn-model-val_0.5339.hdf5 \
    --output_path $OUT3 \
    --predict

echo "[PREDICT] generate $OUT4 from pre-trained model."
python3 tfidf.py \
    --predict data/test_data.csv \
    --model models/tfidf-model_1.pkl \
    --output $OUT4

echo "[PREDICT] generate $OUT5 from pre-trained model."
python3 tfidf.py \
    --predict data/test_data.csv \
    --model models/tfidf-model_2.pkl \
    --output $OUT5

python3 ensemble.py $OUT1 $OUT2 $OUT3 $OUT4 $OUT5 $OUTPUT_PATH

rm -f $OUT1 $OUT2 $OUT3 $OUT4 $OUT5

