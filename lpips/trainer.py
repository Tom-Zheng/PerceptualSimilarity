
from __future__ import absolute_import

import numpy as np
import torch
from torch import nn
from collections import OrderedDict
from torch.autograd import Variable
from scipy.ndimage import zoom
from tqdm import tqdm
import lpips
import os
import matplotlib.pyplot as plt
from PIL import Image
import pdb
import time
import os
import pandas as pd


class Trainer():
    def name(self):
        return self.model_name

    def initialize(self, model='lpips', net='alex', colorspace='Lab', pnet_rand=False, pnet_tune=False, model_path=None,
            use_gpu=True, printNet=False, spatial=False, 
            is_train=False, lr=.00001, beta1=0.5, version='0.1', gpu_ids=[0]):
        '''
        INPUTS
            model - ['lpips'] for linearly calibrated network
                    ['baseline'] for off-the-shelf network
                    ['L2'] for L2 distance in Lab colorspace
                    ['SSIM'] for ssim in RGB colorspace
            net - ['squeeze','alex','vgg']
            model_path - if None, will look in weights/[NET_NAME].pth
            colorspace - ['Lab','RGB'] colorspace to use for L2 and SSIM
            use_gpu - bool - whether or not to use a GPU
            printNet - bool - whether or not to print network architecture out
            spatial - bool - whether to output an array containing varying distances across spatial dimensions
            is_train - bool - [True] for training mode
            lr - float - initial learning rate
            beta1 - float - initial momentum term for adam
            version - 0.1 for latest, 0.0 was original (with a bug)
            gpu_ids - int array - [0] by default, gpus to use
        '''
        self.use_gpu = use_gpu
        self.gpu_ids = gpu_ids
        self.model = model
        self.net = net
        self.is_train = is_train
        self.spatial = spatial
        self.model_name = '%s [%s]'%(model,net)

        if(self.model == 'lpips'): # pretrained net + linear layer
            self.net = lpips.LPIPS(pretrained=not is_train, net=net, version=version, lpips=True, spatial=spatial, 
                pnet_rand=pnet_rand, pnet_tune=pnet_tune, 
                use_dropout=True, model_path=model_path, eval_mode=False)
        elif(self.model=='baseline'): # pretrained network
            self.net = lpips.LPIPS(pnet_rand=pnet_rand, net=net, lpips=False)
        elif(self.model in ['L2','l2']):
            self.net = lpips.L2(use_gpu=use_gpu,colorspace=colorspace) # not really a network, only for testing
            self.model_name = 'L2'
        elif(self.model in ['DSSIM','dssim','SSIM','ssim']):
            self.net = lpips.DSSIM(use_gpu=use_gpu,colorspace=colorspace)
            self.model_name = 'SSIM'
        else:
            raise ValueError("Model [%s] not recognized." % self.model)

        self.parameters = list(self.net.parameters())

        if self.is_train: # training mode
            # extra network on top to go from distances (d0,d1) => predicted human judgment (h*)
            self.rankLoss = lpips.BCERankingLoss()
            self.parameters += list(self.rankLoss.net.parameters())
            self.lr = lr
            self.old_lr = lr
            self.optimizer_net = torch.optim.Adam(self.parameters, lr=lr, betas=(beta1, 0.999))
        else: # test mode
            self.net.eval()

        if(use_gpu):
            self.net.to(gpu_ids[0])
            self.net = torch.nn.DataParallel(self.net, device_ids=gpu_ids)
            if(self.is_train):
                self.rankLoss = self.rankLoss.to(device=gpu_ids[0]) # just put this on GPU0

        if(printNet):
            print('---------- Networks initialized -------------')
            networks.print_network(self.net)
            print('-----------------------------------------------')

    def forward(self, in0, in1, retPerLayer=False):
        ''' Function computes the distance between image patches in0 and in1
        INPUTS
            in0, in1 - torch.Tensor object of shape Nx3xXxY - image patch scaled to [-1,1]
        OUTPUT
            computed distances between in0 and in1
        '''

        return self.net.forward(in0, in1, retPerLayer=retPerLayer)

    # ***** TRAINING FUNCTIONS *****
    def optimize_parameters(self):
        self.forward_train()
        self.optimizer_net.zero_grad()
        self.backward_train()
        self.optimizer_net.step()
        self.clamp_weights()

    def clamp_weights(self):
        for module in self.net.modules():
            if(hasattr(module, 'weight') and module.kernel_size==(1,1)):
                module.weight.data = torch.clamp(module.weight.data,min=0)

    def set_input(self, data):
        self.input_ref = data['ref']
        self.input_p0 = data['p0']
        self.input_p1 = data['p1']
        self.input_judge = data['judge']

        if(self.use_gpu):
            self.input_ref = self.input_ref.to(device=self.gpu_ids[0])
            self.input_p0 = self.input_p0.to(device=self.gpu_ids[0])
            self.input_p1 = self.input_p1.to(device=self.gpu_ids[0])
            self.input_judge = self.input_judge.to(device=self.gpu_ids[0])

        self.var_ref = Variable(self.input_ref,requires_grad=True)
        self.var_p0 = Variable(self.input_p0,requires_grad=True)
        self.var_p1 = Variable(self.input_p1,requires_grad=True)

    def forward_train(self): # run forward pass
        self.d0 = self.forward(self.var_ref, self.var_p0)
        self.d1 = self.forward(self.var_ref, self.var_p1)
        self.acc_r = self.compute_accuracy(self.d0,self.d1,self.input_judge)

        self.var_judge = Variable(1.*self.input_judge).view(self.d0.size())

        self.loss_total = self.rankLoss.forward(self.d0, self.d1, self.var_judge*2.-1.)

        return self.loss_total

    def backward_train(self):
        torch.mean(self.loss_total).backward()

    def compute_accuracy(self,d0,d1,judge):
        ''' d0, d1 are Variables, judge is a Tensor '''
        d1_lt_d0 = (d1<d0).cpu().data.numpy().flatten()
        judge_per = judge.cpu().numpy().flatten()
        return d1_lt_d0*judge_per + (1-d1_lt_d0)*(1-judge_per)

    def get_current_errors(self):
        retDict = OrderedDict([('loss_total', self.loss_total.data.cpu().numpy()),
                            ('acc_r', self.acc_r)])

        for key in retDict.keys():
            retDict[key] = np.mean(retDict[key])

        return retDict

    def get_current_visuals(self):
        zoom_factor = 256/self.var_ref.data.size()[2]

        ref_img = lpips.tensor2im(self.var_ref.data)
        p0_img = lpips.tensor2im(self.var_p0.data)
        p1_img = lpips.tensor2im(self.var_p1.data)

        ref_img_vis = zoom(ref_img,[zoom_factor, zoom_factor, 1],order=0)
        p0_img_vis = zoom(p0_img,[zoom_factor, zoom_factor, 1],order=0)
        p1_img_vis = zoom(p1_img,[zoom_factor, zoom_factor, 1],order=0)

        return OrderedDict([('ref', ref_img_vis),
                            ('p0', p0_img_vis),
                            ('p1', p1_img_vis)])

    def save(self, path, label):
        if(self.use_gpu):
            self.save_network(self.net.module, path, '', label)
        else:
            self.save_network(self.net, path, '', label)
        self.save_network(self.rankLoss.net, path, 'rank', label)

    # helper saving function that can be used by subclasses
    def save_network(self, network, path, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(path, save_filename)
        torch.save(network.state_dict(), save_path)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        print('Loading network from %s'%save_path)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self,nepoch_decay):
        lrd = self.lr / nepoch_decay
        lr = self.old_lr - lrd

        for param_group in self.optimizer_net.param_groups:
            param_group['lr'] = lr

        print('update lr [%s] decay: %f -> %f' % (type,self.old_lr, lr))
        self.old_lr = lr


    def get_image_paths(self):
        return self.image_paths

    def save_done(self, flag=False):
        np.save(os.path.join(self.save_dir, 'done_flag'),flag)
        np.savetxt(os.path.join(self.save_dir, 'done_flag'),[flag,],fmt='%i')

    def set_eval(self):
        self.net.eval()
    
    def set_train(self):
        self.net.train()

def score_2afc_dataset(data_loader, func, name=''):
    ''' Function computes Two Alternative Forced Choice (2AFC) score using
        distance function 'func' in dataset 'data_loader'
    INPUTS
        data_loader - CustomDatasetDataLoader object - contains a TwoAFCDataset inside
        func - callable distance function - calling d=func(in0,in1) should take 2
            pytorch tensors with shape Nx3xXxY, and return numpy array of length N
    OUTPUTS
        [0] - 2AFC score in [0,1], fraction of time func agrees with human evaluators
        [1] - dictionary with following elements
            d0s,d1s - N arrays containing distances between reference patch to perturbed patches 
            gts - N array in [0,1], preferred patch selected by human evaluators
                (closer to "0" for left patch p0, "1" for right patch p1,
                "0.6" means 60pct people preferred right patch, 40pct preferred left)
            scores - N array in [0,1], corresponding to what percentage function agreed with humans
    CONSTS
        N - number of test triplets in data_loader
    '''

    d0s = []
    d1s = []
    gts = []

    for data in tqdm(data_loader.load_data(), desc=name):
        d0s+=func(data['ref'],data['p0']).data.cpu().numpy().flatten().tolist()
        d1s+=func(data['ref'],data['p1']).data.cpu().numpy().flatten().tolist()
        gts+=data['judge'].cpu().numpy().flatten().tolist()

    d0s = np.array(d0s)
    d1s = np.array(d1s)
    gts = np.array(gts)
    scores = (d0s<d1s)*(1.-gts) + (d1s<d0s)*gts + (d1s==d0s)*.5

    return(np.mean(scores), dict(d0s=d0s,d1s=d1s,gts=gts,scores=scores))

def score_jnd_dataset(data_loader, func, name=''):
    ''' Function computes JND score using distance function 'func' in dataset 'data_loader'
    INPUTS
        data_loader - CustomDatasetDataLoader object - contains a JNDDataset inside
        func - callable distance function - calling d=func(in0,in1) should take 2
            pytorch tensors with shape Nx3xXxY, and return pytorch array of length N
    OUTPUTS
        [0] - JND score in [0,1], mAP score (area under precision-recall curve)
        [1] - dictionary with following elements
            ds - N array containing distances between two patches shown to human evaluator
            sames - N array containing fraction of people who thought the two patches were identical
    CONSTS
        N - number of test triplets in data_loader
    '''

    ds = []
    gts = []

    for data in tqdm(data_loader.load_data(), desc=name):
        ds+=func(data['p0'],data['p1']).data.cpu().numpy().tolist()
        gts+=data['same'].cpu().numpy().flatten().tolist()

    sames = np.array(gts)
    ds = np.array(ds)

    sorted_inds = np.argsort(ds)
    ds_sorted = ds[sorted_inds]
    sames_sorted = sames[sorted_inds]

    TPs = np.cumsum(sames_sorted)
    FPs = np.cumsum(1-sames_sorted)
    FNs = np.sum(sames_sorted)-TPs

    precs = TPs/(TPs+FPs)
    recs = TPs/(TPs+FNs)
    score = lpips.voc_ap(recs,precs)

    return(score, dict(ds=ds,sames=sames))


def score_tnn_dataset(data_loader, func, epoch=0, name=''):
    ''' Function computes Two Alternative Forced Choice (2AFC) score using
        distance function 'func' in dataset 'data_loader'
    INPUTS
        data_loader - CustomDatasetDataLoader object - contains a TwoAFCDataset inside
        func - callable distance function - calling d=func(in0,in1) should take 2
            pytorch tensors with shape Nx3xXxY, and return numpy array of length N
    OUTPUTS
        [0] - 2AFC score in [0,1], fraction of time func agrees with human evaluators
        [1] - dictionary with following elements
            d0s,d1s - N arrays containing distances between reference patch to perturbed patches 
            gts - N array in [0,1], preferred patch selected by human evaluators
                (closer to "0" for left patch p0, "1" for right patch p1,
                "0.6" means 60pct people preferred right patch, 40pct preferred left)
            scores - N array in [0,1], corresponding to what percentage function agreed with humans
    CONSTS
        N - number of test triplets in data_loader
    '''

    d0s = []
    d1s = []
    gts = []

    ref_paths = []
    p0_paths = []
    p1_paths = []

    for data in tqdm(data_loader.load_data(), desc=name):
        d0s+=func(data['ref'],data['p0']).data.cpu().numpy().flatten().tolist()
        d1s+=func(data['ref'],data['p1']).data.cpu().numpy().flatten().tolist()
        gts+=data['judge'].cpu().numpy().flatten().tolist()

        ref_paths += data['ref_path']
        p0_paths += data['p0_path']
        p1_paths += data['p1_path']


    d0s = np.array(d0s)
    d1s = np.array(d1s)
    gts = np.array(gts)


    # if epoch > 17:
    if False:
    # if epoch <= 1 or epoch > 17:

        print(d0s)
        # pdb.set_trace()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(d0s.shape[0]), d0s)
        ax.plot(range(d1s.shape[0]), d1s)
        plt.title("d")

        fig = plt.figure()
        n_selected = 4
        selected = np.random.choice(len(ref_paths), n_selected , replace=False)
        for i in range(n_selected):
            ax_ref = fig.add_subplot(4,3,i*3+1)
            ax_p0 = fig.add_subplot(4,3,i*3+2)
            ax_p1 = fig.add_subplot(4,3,i*3+3)
            ax_ref.imshow(Image.open(ref_paths[selected[i]]).convert('RGB'))
            ax_p0.imshow(Image.open(p0_paths[selected[i]]).convert('RGB'))
            ax_p1.imshow(Image.open(p1_paths[selected[i]]).convert('RGB'))
            print(ref_paths[i])
            print(p0_paths[i])
            print(p1_paths[i])

            print("sample i: {}; d0: {}; d1: {}; gt: {}".format(selected[i], d0s[selected[i]], d1s[selected[i]], gts[selected[i]]))
            
        plt.show()

        

    scores = (d0s<d1s)*(1.-gts) + (d1s<d0s)*gts + (d1s==d0s)*.5

    print('gts ratio: ', gts.sum() / gts.shape[0] )

    return(np.mean(scores), dict(d0s=d0s,d1s=d1s,gts=gts,scores=scores))

def eval_tnn_testset(data_loader, func, save_path):
    ''' Function computes Two Alternative Forced Choice (2AFC) score using
        the dataloader will iterate over lr-ref pairs (lr, ref_i, _not_used)
        N lrs, M refs

        output: NxM csv, filled with distances 
    '''
    d0s = []
    ref_paths = []   #lr
    p0_paths = []    #ref
    
    start = time.time()
    for data in tqdm(data_loader.load_data(), desc='test'):
        d0s+=func(data['ref'],data['p0']).data.cpu().numpy().flatten().tolist()
        ref_paths += data['ref_path']
        p0_paths += data['p0_path']
    end = time.time()

    ref_paths_unique = sorted(list(set(ref_paths)))
    p0_paths_unique = sorted(list(set(p0_paths)))

    N = len(ref_paths_unique)
    M = len(p0_paths_unique)

    assert(len(d0s) == N*M)
    scores = np.array(d0s).reshape((N,M))

    scores_df = pd.DataFrame(data=scores,
                            index=ref_paths_unique, 
                            columns=p0_paths_unique)
    
    path = os.path.join(save_path,'latest_lpips_test.csv')
    scores_df.to_csv(path)

    print("Evaluated {} pairs in {} seconds.".format(len(d0s), end-start))
    print('Save to:', path)

def eval_tnn_testset_psnr(data_loader, func, save_path, epoch, psnrs ,name="lpips_test.csv"):
    ''' Function computes Two Alternative Forced Choice (2AFC) score using
        the dataloader will iterate over lr-ref pairs (lr, ref_i, _not_used)
        N lrs, M refs

        output: NxM csv, filled with distances 
    '''
    d0s = []
    ref_paths = []   #lr
    p0_paths = []    #ref
    
    start = time.time()
    for data in tqdm(data_loader.load_data(), desc='test'):
        d0s+=func(data['ref'],data['p0']).data.cpu().numpy().flatten().tolist()
        ref_paths += data['ref_path']
        p0_paths += data['p0_path']
    end = time.time()

    ref_paths_unique = sorted(list(set(ref_paths)))
    p0_paths_unique = sorted(list(set(p0_paths)))

    N = len(ref_paths_unique)
    M = len(p0_paths_unique)

    assert(len(d0s) == N*M)
    scores = np.array(d0s).reshape((N,M))

    scores_df = pd.DataFrame(data=scores,
                            index=ref_paths_unique, 
                            columns=p0_paths_unique)
    
    path = os.path.join(save_path, 'epoch_{:03d}_'.format(epoch)+name)
    scores_df.to_csv(path)

    print("Evaluated {} pairs in {} seconds.".format(len(d0s), end-start))
    print('Save to:', path)

    # compute avg psnr...
    # psnr_df = pd.read_csv(os.path.join(base_exp,'psnr.csv'))
    # pdb.set_trace()
    test_psnrs = psnrs[-N:]
    assert(test_psnrs.shape == scores.shape)
    best_psnr = test_psnrs.max(axis=1).mean()

    random_indices = np.random.randint(test_psnrs.shape[1],size=test_psnrs.shape[0])
    random_psnr = test_psnrs[range(test_psnrs.shape[0]), random_indices].mean()
    # compute random PSNR 

    # compute selected PSNR
    ind = np.argsort(scores, axis=1)

    ind_i = np.tile(np.arange(N).reshape(N,1),M)
    ours_psnrs = test_psnrs[ind_i,ind]
    lpips_psnr = ours_psnrs[:,0].mean()

    return {'best_psnr': best_psnr,
            'random_psnr': random_psnr,
            'lpips_psnr': lpips_psnr}
