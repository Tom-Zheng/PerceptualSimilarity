import numpy as np
import pandas as pd
import os.path
import pdb

ROOT = "/home/zheng/Desktop/rl/data/20201023_small_training/hr_few_person_7_mode_3/7/170"
df = pd.read_csv(os.path.join(ROOT, "mapping.csv"))

lr_list = df['lr_path'].unique()

pivot = len(lr_list) // 2 + 1
train_df = df[df['lr_path'] < lr_list[pivot]]
val_df = df[df['lr_path'] >= lr_list[pivot]]

print(train_df)
print(val_df)
print(len(train_df['lr_path'].unique()))
print(len(val_df['lr_path'].unique()))

# print(train_df.iloc[-1])

# val_df = df.iloc[820:]
# process 

# for i in val_df.index:
#     ref_path = val_df['ref_path'][i]
#     if (ref_path[-8:-4] < "0755"):
#         val_df = val_df.drop(i)

# print(val_df['ref_path'])

train_df.to_csv(os.path.join(ROOT, "train.csv"))
val_df.to_csv(os.path.join(ROOT, "val.csv"))