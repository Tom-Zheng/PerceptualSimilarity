import numpy as np
import pandas as pd
import os.path
import pdb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, help='base path for images')
parser.add_argument('--csv_path', type=str, help='base path for csv and result images')
parser.add_argument('--lpips_train_ratio', type=float, help='training ratio (lpips)')
parser.add_argument('--test_ratio', type=float, help='test ratio')

args = parser.parse_args()

# | -----train1----- |- train2 | --val---| xxxxxxxxxxxxxx|----- test------|
df = pd.read_csv(os.path.join(args.csv_path,'psnr.csv'))
ref_paths = df.columns[1:]
rest_paths = df['Unnamed: 0'].values.tolist()
n_train1 = len(ref_paths)
n_rest = len(rest_paths)
n_total = n_train1 + n_rest
n_train2 = int(n_train1 * args.lpips_train_ratio) 
n_test = int(n_total * args.test_ratio)
n_val = n_train1 - n_train2

assert(n_val > 0)

df_mapping = pd.read_csv(os.path.join(args.csv_path,'mapping.csv'))
lr_list = df_mapping['lr_path'].unique()
training_lr_list = lr_list[:n_train2]
val_lr_list = lr_list[n_train2:n_train2 + n_val]
test_lr_list = lr_list[-n_test:]

train_df = df_mapping.loc[df_mapping['lr_path'].isin(training_lr_list)]
val_df = df_mapping.loc[df_mapping['lr_path'].isin(val_lr_list)]
test_df = df_mapping.loc[df_mapping['lr_path'].isin(test_lr_list)]

print(train_df)
print(val_df)
print(test_df)

print(len(train_df['lr_path'].unique()))
print(len(val_df['lr_path'].unique()))
print(len(test_df['lr_path'].unique()))

# print(train_df.iloc[-1])

# val_df = df.iloc[820:]
# process 

# for i in val_df.index:
#     ref_path = val_df['ref_path'][i]
#     if (ref_path[-8:-4] < "0755"):
#         val_df = val_df.drop(i)

# print(val_df['ref_path'])

train_df.to_csv(os.path.join(args.csv_path, "train.csv"))
val_df.to_csv(os.path.join(args.csv_path, "val.csv"))
test_df.to_csv(os.path.join(args.csv_path, "test.csv"))

lst = [[n_train1, n_train2, n_val, n_test, n_total]] 
df_count = pd.DataFrame(lst, columns =['train1', 'train2','val','test','total'])
df_count.to_csv(os.path.join(args.csv_path, "dataset_sizes.csv"))
