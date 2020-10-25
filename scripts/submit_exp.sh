#!/bin/bash

TRIAL=${1}
IMG_PATH=${2}
ORI_CSV_PATH=${3}
GPU_ID=${4}
EXP_ID=${5}

NET=alex

mkdir checkpoints
mkdir checkpoints/${TRIAL}
mkdir checkpoints/${TRIAL}/output

CSV_PATH=checkpoints/${TRIAL}/output

cp ${ORI_CSV_PATH}/*.csv ${CSV_PATH}

python process_train_val.py \
--img_path  ${IMG_PATH} \
--csv_path  ${CSV_PATH} \
--lpips_train_ratio  0.8 \
--test_ratio 0.2 

python ./train.py --use_gpu --net ${NET} --name ${TRIAL} \
--nepoch 1 \
--nepoch_decay 1 \
--dataset_mode tnn \
--display_id $(( 1 + 4*${EXP_ID} )) \
--print_freq 10 \
--display_freq 10 \
--batch_size 64 \
--load_size 64 \
--train_trunk \
--train_plot \
--gpu_ids ${GPU_ID} \
--img_path ${IMG_PATH} \
--csv_path ${CSV_PATH} 

# --from_scratch \

python ./test_dataset_model.py --use_gpu --net ${NET} \
--model_path ./checkpoints/${TRIAL}/latest_net_.pth  \
--dataset_mode tnn \
--datasets test \
--load_size 64 \
--batch_size 64 \
--train_trunk \
--img_path ${IMG_PATH} \
--csv_path ${CSV_PATH} 
