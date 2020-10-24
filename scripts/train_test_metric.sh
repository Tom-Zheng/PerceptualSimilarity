
TRIAL=${1}
NET=${2}
mkdir checkpoints
mkdir checkpoints/${NET}_${TRIAL}
python ./train.py --use_gpu --net ${NET} --name ${NET}_${TRIAL} \
--nepoch 1000 \
--nepoch_decay 1000 \
--dataset_mode tnn \
--display_id 17 \
--print_freq 10 \
--display_freq 10 \
--batch_size 80 \
--load_size 64 \
--train_trunk \
--train_plot  

# --from_scratch \

# python ./test_dataset_model.py --use_gpu --net ${NET} --model_path ./checkpoints/${NET}_${TRIAL}/latest_net_.pth \ 
# --train_trunk
