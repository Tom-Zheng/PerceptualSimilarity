#!/bin/bash

IMG_PATH=${1}
CSV_PATH=${2}
TR=${3}

python process_train_val.py \
--img_path  ${IMG_PATH} \
--csv_path  ${CSV_PATH} \
--total_training_ratio  ${TR} \
--lpips_train_ratio  0.8 \
--test_ratio 0.2 

# --img_path  /home/zheng/Desktop/rl/data/20201024/img \
# --csv_path  /home/zheng/Desktop/rl/data/20201024/exp211_False_False \