import os.path
import torchvision.transforms as transforms
from data.dataset.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
import pdb
import pandas as pd
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pickle
import os
# from IPython import embed

class TNNDataset(BaseDataset):
    def initialize(self, dataroots, load_size=256, opt=None):

        csv_path = None
        self.test = False

        if './dataset/tnn/val' in dataroots: # val
            csv_path = os.path.join(opt.csv_path, 'val.csv')
            print('loading val: ', csv_path)
        elif './dataset/tnn/test' in dataroots: # test
            csv_path = os.path.join(opt.csv_path, 'test.csv')
            self.test = True
            print('loading test: ', csv_path)
        else:
            csv_path = os.path.join(opt.csv_path, 'train.csv')
            print('loading train: ', csv_path)

        self.association = pd.read_csv(csv_path) # ['Unnamed: 0', 'hr_path', 'lr_path', 'psnr', 'ref_path', 'sr_path','ssim'],
        self.root = opt.img_path
        self.load_size = load_size

        self.ref_paths = self.association['lr_path'].unique()

        N = len(self.ref_paths)
        M = len(self.association['ref_path'].unique())


        if self.test:
            assert(len(self.association) == N*M) # sanity check
            self.N_samples = len(self.association)
        else:
            self.N_samples = len(self.ref_paths)

        # self.dir_p1 = [os.path.join(root, 'p1') for root in self.roots]
        # self.p1_paths = make_dataset(self.dir_p1)
        # self.p1_paths = sorted(self.p1_paths)

        transform_list = []
        transform_list.append(transforms.Scale(load_size))
        transform_list += [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

        # judgement directory
        # self.dir_J = [os.path.join(root, 'judge') for root in self.roots]
        # self.judge_paths = make_dataset(self.dir_J,mode='np')
        # self.judge_paths = sorted(self.judge_paths)

    def __getitem__(self, index):
        if not self.test:
            ref_path = os.path.join(self.root, self.ref_paths[index])
            ref_img_ = Image.open(ref_path).convert('RGB')
            ref_img = self.transform(ref_img_)

            df_p0_p1 = self.association[self.association['lr_path'] == self.ref_paths[index]]
            # print(self.ref_paths[index])
            # print(df_p0_p1[['ref_path', 'psnr']])
            N_associated = len(df_p0_p1)

            indices = torch.randperm(N_associated)[:2]     # numpy choice is a bad idea for dataloader
            i = indices[0].item()
            j = indices[1].item()

            # pdb.set_trace()
            p0_path = os.path.join(self.root, df_p0_p1.iloc[i]['ref_path'])
            p0_img_ = Image.open(p0_path).convert('RGB')
            p0_img = self.transform(p0_img_)

            p1_path = os.path.join(self.root, df_p0_p1.iloc[j]['ref_path'])
            p1_img_ = Image.open(p1_path).convert('RGB')
            p1_img = self.transform(p1_img_)

            # p1 is more similar than p0

            judge_img = np.array([float(df_p0_p1.iloc[j]['psnr'] > df_p0_p1.iloc[i]['psnr'])]).reshape((1,1,1,)) # [0,1]
            judge_img = torch.FloatTensor(judge_img)
        
        # for test set
        else:
            ref_path = os.path.join(self.root, self.association.loc[index]['lr_path'])
            ref_img_ = Image.open(ref_path).convert('RGB')
            ref_img = self.transform(ref_img_)

            p0_path = os.path.join(self.root, self.association.loc[index]['ref_path'])
            p0_img_ = Image.open(p0_path).convert('RGB')
            p0_img = self.transform(p0_img_)

            # not used
            p1_path = p0_path
            p1_img = p0_img
            judge_img = np.array([0.0]).reshape((1,1,1,)) # [0,1]
            judge_img = torch.FloatTensor(judge_img)

        return {'p0': p0_img, 'p1': p1_img, 'ref': ref_img, 'judge': judge_img,
            'p0_path': p0_path, 'p1_path': p1_path, 'ref_path': ref_path}

    def __len__(self):
        return self.N_samples

if __name__ == '__main__':
    class Opt:
        def __init__(self):
            self.img_path = '/home/zheng/Desktop/rl/data/20201024/img'
            self.csv_path = '/home/zheng/Desktop/rl/data/20201024/exp211_False_False'

    opt = Opt()

    # np.random.seed(1234)
    dataset = TNNDataset()
    roots_path = ['./dataset/tnn/val']
    dataset.initialize(roots_path, load_size=64, opt=opt)

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=1)
    print(len(dataset), len(dataloader))

    for i_batch, sample_batched in enumerate(dataloader):
        # print(i_batch, sample_batched['gt'].size())
        # visualization
        print(sample_batched['ref_path'])
        print(sample_batched['p0_path'])
        images_batch = sample_batched['ref']-sample_batched['ref'][:1,:,:,:]
        batch_size = images_batch.size()[0]
        im_size = images_batch.size()[1:]
        print(batch_size, im_size)
        grid = utils.make_grid(images_batch, nrow=8)
        plt.imshow(grid.numpy().transpose(1, 2, 0))
        plt.show()
        grid = utils.make_grid(sample_batched['p0'], nrow=8)
        plt.imshow(grid.numpy().transpose(1, 2, 0))
        plt.show()

