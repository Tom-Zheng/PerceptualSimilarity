import numpy as np
import pandas as pd
import os.path
import pdb

ROOT = "/home/zheng/Desktop/rl/data/20201022/hr_few_person_7_lr_bicubic_x8_crossnetpp_train_window_size20_gen_window_size_50/7/170_test"
df = pd.read_csv(os.path.join(ROOT, "mapping.csv"))
# print(df)

train_df = df.iloc[:820]
# print(train_df.iloc[-1])

val_df = df.iloc[820:]
# process 

for i in val_df.index:
    ref_path = val_df['ref_path'][i]
    if (ref_path[-8:-4] < "0755"):
        val_df = val_df.drop(i)

# print(val_df['ref_path'])

train_df.to_csv(os.path.join(ROOT, "train.csv"))
val_df.to_csv(os.path.join(ROOT, "val.csv"))