#!/bin/bash

MOD_DIR=models
dst_model="$MOD_DIR/model-0.7073.hdf5"
src_model_a="$MOD_DIR/split-model-a"
src_model_b="$MOD_DIR/split-model-b"

if [ ! -f $dst_model ]; then
    echo "$dst_model doesn't exist!"
    
    if [[ -f $src_model_a && -f $src_model_b ]]; then
        echo "splitted models: $src_model_a and $src_model_b exist."
        echo "merging $src_model_a and $src_model_b"
        
        touch $dst_model
        cat $src_model_a >> $dst_model
        cat $src_model_b >> $dst_model
        echo "created $dst_model"
    else
        echo "missing $src_model_a and $src_model_b!"
    fi
fi

# python version: python3.5
python3 ensemble_predict.py $1 $2 $MOD_DIR/model-0.6983.hdf5 $MOD_DIR/model-0.6990.hdf5 $MOD_DIR/model-0.7047.hdf5 $MOD_DIR/model-0.7057.hdf5 $MOD_DIR/model-0.7073.hdf5
