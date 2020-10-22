
TRIAL=${1}
NET=${2}
mkdir checkpoints
mkdir checkpoints/${NET}_${TRIAL}
python ./train.py --use_gpu --net ${NET} --name ${NET}_${TRIAL} \
--nepoch 200 \
--nepoch_decay 200 \
--display_freq 10 \
--print_freq 10 \
--dataset_mode tnn \
--batch_size 64 \
--datasets train/traditional \
--train_plot  
# --train_trunk

# python ./test_dataset_model.py --use_gpu --net ${NET} --model_path ./checkpoints/${NET}_${TRIAL}/latest_net_.pth \ 
# --train_trunk
