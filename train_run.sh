#!/bin/bash
# First argument is gpu number
# Second argument is gpu name

if [ -z "$1" ]
    then
        echo "Please enter dataset name as first argument (e.g., beauty, mobile, fashion)"
        exit 1
fi

if [ -z "$2" ]
    then
        echo "Please enter gpu number as second argument!"
        exit 1
fi

if [ -z "$3" ]
    then
        echo "Please enter gpu name as third argument"
        exit 1
fi

start_idx="0"
end_idx="20"

if [ -z "$4" ]
    then
        echo "Warning! No start index specified, Using default start index:"
        echo $start_idx
else
    start_idx=$4
fi

if [ -z "$5" ]
    then
        echo "Warning! No end index specified, Using default end index:"
        echo $end_idx
else
    end_idx=$5
fi

dataset_name=$1
gpu_num=$2
gpu_name=$3

optimizer="adam"
epochs="15"
model_type="rescnn"
img_dim="640"

dataset_path="./data"

CUDA_VISIBLE_DEVICES=${gpu_num} python3 train.py \
--out-dir ${dataset_path}/output/${dataset_name}-${optimizer}-mt${model_type}-imgdim${img_dim}-ep${epochs}-s${start_idx}-e${end_idx}-${gpu_name} \
--train-csv-path ${dataset_path}/ndsc-advanced/${dataset_name}_data_info_train_competition.csv \
--final-test-csv-path ${dataset_path}/ndsc-advanced/${dataset_name}_data_info_val_competition.csv \
--img-path ${dataset_path} --img-dim ${img_dim} \
--optimizer ${optimizer} --model-type ${model_type} \
--start-class-index ${start_idx} --end-class-index ${end_idx} \
-b 6 -be 6 --epochs ${epochs}

