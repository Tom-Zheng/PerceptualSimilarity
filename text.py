from torchvision import transforms, utils
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pickle
import os
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)
print(len(dataset), len(dataloader))

img_name = ['img1_LR', 'img1_SR', 'img1_HR', 'img2_HR']
for i_batch, sample_batched in enumerate(dataloader):
    # print(i_batch, sample_batched['gt'].size())
    # visualization
    images_batch = sample_batched['input_img1_LR']
    batch_size = images_batch.size()[0]
    im_size = images_batch.size()[1:]
    print(batch_size, im_size)
    grid = utils.make_grid(images_batch, nrow=8)
    plt.imshow(grid.numpy().transpose(1, 2, 0))
    plt.show()
    grid = utils.make_grid(sample_batched['input_img1_HR'], nrow=8)
    plt.imshow(grid.numpy().transpose(1, 2, 0))
    plt.show()