
TRIAL=${1}
NET=${2}
mkdir checkpoints
mkdir checkpoints/${NET}_${TRIAL}_tune
python ./train.py --use_gpu --net ${NET} --name ${NET}_${TRIAL}_tune \
--display_id 1 \
--print_freq 10 \
--load_size 64 \
--batch_size 256 \
--dataset_mode 2afc

python ./test_dataset_model.py --train_trunk --use_gpu --net ${NET} --model_path ./checkpoints/${NET}_${TRIAL}_tune/latest_net_.pth
