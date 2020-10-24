
TRIAL=${1}
NET=${2}
mkdir checkpoints
mkdir checkpoints/${NET}_${TRIAL}
python ./train.py --use_gpu --net ${NET} --name ${NET}_${TRIAL} \
--nepoch 500 \
--nepoch_decay 500 \
--dataset_mode tnn \
--display_id 10 \
--print_freq 10 \
--batch_size 16 \
--load_size 128 \
--train_trunk \
--train_plot  
# --display_freq 10 \
# --from_scratch \


# python ./test_dataset_model.py --use_gpu --net ${NET} --model_path ./checkpoints/${NET}_${TRIAL}/latest_net_.pth \ 
# --train_trunk
