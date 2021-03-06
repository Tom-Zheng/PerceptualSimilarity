TRIAL=${1}
NET=${2}
GPU_ID=${3}
EXP_ID=${4}
IMG_PATH=${5}
CSV_PATH=${6}

mkdir checkpoints
mkdir checkpoints/${NET}_${TRIAL}
python ./train.py --use_gpu --net ${NET} --name ${NET}_${TRIAL} \
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
--model_path ./checkpoints/${NET}_${TRIAL}/latest_net_.pth  \
--dataset_mode tnn \
--datasets test \
--load_size 64 \
--batch_size 64 \
--train_trunk \
--img_path ${IMG_PATH} \
--csv_path ${CSV_PATH} 
