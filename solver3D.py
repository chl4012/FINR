import os
import json
import time
import scipy.io as sio
import mat73
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from utils.common_utils import *
import torchkbnufft as tkbn
from torch.fft import fftshift, fft, ifft
from utils.loss import ncc_loss
import h5py
import matplotlib.pyplot as plt
from matplotlib import animation

from IPython.core.debugger import set_trace

from utils.nufft_utils import *
from Mypnufft_mc_func_liver_MRF import *
from Mypnufft_mc_func_liver_MGRE import *
from Mypnufft_mc_func_liver_MRF_3D import *
from models.unetVgg import unetVgg 
from models.unet2D import *
from sdf.provider import *
from sdf.utils import *


class Solver_MRF_fat_water_HashGrid3D_STN():
    def __init__(self, module, opt):        
        self.opt = opt
        self.prepare_dataset()        
        self.nslice = self.Nslice
        opt.nslice = self.nslice
        Ur = torch.tensor(self.Ur)
        dc = torch.tensor(self.kdens)
        coilmap = torch.tensor(self.coilmap)
        self.sli = opt.slice
        par = self.par
        T1_stacks = par['T1_stacks']
        MGRE_stacks_ = par['MGRE_stacks_']
        MGRE_stacks = par['MGRE_stacks']
        T2_stacks = par['T2_stacks']
        nslice_se = par['nslice_se']
        nslice_me = par['nslice_me']
        Ur = par['Ur']
        dfat = par['dfat']
        te = par['te']
        Nrep = self.Nrep
        if Ur.shape[0] == T1_stacks + MGRE_stacks_ + T2_stacks:
            self.Ur_full_list = get_Ur_list(T1_stacks, MGRE_stacks_, T2_stacks, 
                            Nrep, nslice_se, nslice_me, Ur)
        else:
            self.Ur_full_list = get_Ur_list_v2(T1_stacks, MGRE_stacks_, T2_stacks, 
                            Nrep, nslice_se, nslice_me, Ur)
        self.ktraj_view_list = get_ktraj_view_list( T1_stacks, MGRE_stacks, T2_stacks,
                              Nrep, nslice_se) #size(32000,)
        self.slice_dict_MRF = self.slice_dict_MRF
        

        from models.netowrk import SDFNetwork3D_STN_MRF_fat_water

        self.model = SDFNetwork3D_STN_MRF_fat_water(dc, coilmap, par, opt= self.opt, dfat = dfat, te = te)
        print("current device is: " + str(self.dev))
        
        self.T_ = self.T.transpose()
        self.index_all = self.index_all
        opt = self.opt



    def fit(self):
        opt = self.opt
        train_dataset = HashGrid3D_MRF_fat_water_Dataset(self.kspaceMRF, self.T_, self.index_all, self.par, self.motion_values, 
                                            self.Ur_full_list, self.slice_dict_MRF, self.ktraj_view_list, size=self.binnum, flat_flag=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
        criterion = torch.nn.L1Loss() # torch.nn.L1Loss()
        scheduler = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, opt.step_size, gamma=opt.gamma)
        optimizer = lambda model: torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.99), eps=1e-15)
        trainer = Trainer('MRF_fat_water_STN', self.model, workspace=opt.ckpt_root,
                          criterion=criterion, ema_decay=0.95, local_rank=0,       
                          use_multi_gpus=opt.use_multi_gpus, gpu_ids=opt.gpu_ids,  
                          fp16=opt.fp16, lr_scheduler=scheduler, optimizer=optimizer,
                          use_checkpoint='latest', eval_interval=1, par=self.par, opt=self.opt)
        trainer.train(train_loader, opt.max_steps)
    
        
    @torch.no_grad()
    def evaluate(self):  
        print("---evaluating models")
        opt = self.opt
        criterion = torch.nn.L1Loss() # torch.nn.L1Loss()
        eval_dataset = HashGrid3D_MRF_fat_water_Dataset(self.kspaceMRF, self.T_, self.index_all, self.par, self.motion_values, 
                                            self.Ur_full_list, self.slice_dict_MRF, self.ktraj_view_list, size=self.binnum)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False)
        trainer = Trainer('MRF_fat_water_STN', self.model, workspace=opt.ckpt_root, criterion=criterion,
                          local_rank=0, use_multi_gpus=opt.use_multi_gpus, gpu_ids=opt.gpu_ids,
                          fp16=opt.fp16, use_checkpoint='latest', eval_interval=1, par=self.par, opt = self.opt)
        trainer.evaluate(eval_loader)


    def prepare_dataset(self): 
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")     
        file_dir = self.opt.file_dir #Chao Li: filename
        fkspace = os.path.join(file_dir,'kspaceMGRE.mat')
        fT = os.path.join(file_dir,"p_par_T.mat")
        fmotionidx = os.path.join(file_dir,"motion_indices.mat")
        kdens_file = os.path.join(file_dir,'kdens.mat')
        lkut_file = os.path.join(file_dir, 'lookup_table.mat')
        Ur_file = os.path.join(file_dir, 'Ur.mat')
        coilmap_file = os.path.join(file_dir, 'coilmap.mat')
        seqpar_file = os.path.join(file_dir, 'seqpar.mat')
        dipole_file = os.path.join(file_dir, 'dipole.mat')
        seqpar_file = sio.loadmat(seqpar_file)
        print("-----loading MRF kspace data-----")
        kspaceMRF = mat73.loadmat(fkspace)["kspaceMGRE"]  #(288      16     total_readout  Necho)
        print("-----finished-----")
        dipole = sio.loadmat(dipole_file)["dipole"]
        T = sio.loadmat(fT)["T"] #[436     total_readout/slice_me]
        p = sio.loadmat(fT)["p"] 
        motion_values = sio.loadmat(fmotionidx)["motion_indices"] #[1 total_readout/slice_me]
        kdens = sio.loadmat(kdens_file)['kdens']
        view_dict_MRF = sio.loadmat(lkut_file)['view_dict_MRF'].flatten() -1 # [1 total_readout]
        slice_dict_MRF = sio.loadmat(lkut_file)['slice_dict_MRF'].flatten() -1 # [1 total_readout]
        coilmap = sio.loadmat(coilmap_file)["coilmap"] #(Ncoil, 256, 256, Nslice)

        Ur = sio.loadmat(Ur_file)['Ur']
        npoints = kspaceMRF.shape[0]
        Ncoil = int(np.shape(kspaceMRF)[1])

        if not self.opt.istest:
            if self.opt.data_usage_ratio != 1:
                r = self.opt.data_usage_ratio
                s1 = int(T.shape[1]*r)
                s2 = int(view_dict_MRF.shape[0]*r)
                T = T[:,:s1]
                motion_values = motion_values[:,:s1]
                view_dict_MRF = view_dict_MRF[:s2]
                slice_dict_MRF = slice_dict_MRF[:s2]
                kspaceMRF = kspaceMRF[:,:,:s2,:]
                print("use {} of the data".format(r))
        
        ##------------------------------------------------------------------------------
        batch_size = 1
        img_size = 256
        Necho = int(seqpar_file["Necho"])
        T1_stacks = int(seqpar_file["T1_stacks"])
        MGRE_stacks = int(seqpar_file["MGRE_stacks"])
        T2_stacks = int(seqpar_file["T2_stacks"])
        nslice_se = int(seqpar_file["nslice_se"])
        nslice_me = int(seqpar_file["nslice_me"])
        Nslice = int(seqpar_file["Nslice"])
        
        esp = float(seqpar_file["esp"])*10**(-3)
        te = np.arange(0, Necho * esp, esp) #+ 0.00054

        dfat =  -447.19 #seqpar_file['dfat'] chemical shift
        MGRE_stacks_ = int(MGRE_stacks*(nslice_se/nslice_me)) #80
        Ncontrast = T1_stacks + MGRE_stacks_ + T2_stacks
        
        te = te[:Necho]
        if self.opt.use_in_phase:
            kspaceMRF = kspaceMRF[:,:,:,::2]
            te = te[::2]
            print(" using in phase echos")
            echo_used = te.shape[0]
            Necho = echo_used 
        else:
            if Necho != self.opt.echo_used:
                Ne = min(self.opt.echo_used, Necho)
                echo_used = Ne
                print('echo used is '+str(echo_used))
                kspaceMRF = kspaceMRF[:,:,:,:echo_used]
                te = te[:echo_used]
                Necho = echo_used   
        if self.opt.Nsingular_used != Ur.shape[1]:
            Ns = min(self.opt.Nsingular_used, Ur.shape[1])
            Ur = Ur[:,:Ns]
                
        par = {}
        par["T1_stacks"] = T1_stacks
        par["T2_stacks"] = T2_stacks
        par["MGRE_stacks"] = MGRE_stacks
        par["nslice_se"] = nslice_se
        par["nslice_me"] = nslice_me
        par["MGRE_stacks_"] = MGRE_stacks_
        par["Nslice"] = Nslice #the number slices of the final reconstructed image
        par["Ncoil"] = Ncoil
        par["img_size"] = img_size
        par["Necho"] = Necho
        par["Ncontrast"] = Ncontrast
        par["Ur"] = Ur
        par["Nsingular"] = Ur.shape[1]  
        par["batch_size"] = 1   # have to be 1
        par["te"] = te
        par["dfat"] = dfat
        par["dipole"] = dipole

        total_readout = np.shape(kspaceMRF)[2]
        nreadout_per_rep= int(nslice_me*MGRE_stacks_+ nslice_se*(T1_stacks+T2_stacks))

        Nrep = int(total_readout/nreadout_per_rep) #number of repitition: total GA spiral arms divided by the number of contrast
        par["Nrep"] = Nrep
        window_size = self.opt.window_size 
        kdens = kdens/np.mean(kdens)
        
        if self.opt.istest:
            index_all, index_all_2 = get_index_all_non_overlap(motion_values, Nrep, Ncontrast, window_size=window_size)
        else:
            index_all, index_all_2 = get_index_all(motion_values, Nrep, Ncontrast, window_size=window_size)

        binnum = index_all_2.shape[-1]
        motion_values = motion_values.flatten()
        motion_values = motion_values[index_all_2]
        
        self.Ur = Ur
        self.par = par
        self.index_all = index_all
        self.motion_values = motion_values
        self.kdens = kdens
        fov = int(p["fov"])
        T = T*fov/128*pi
        self.T = T
        self.img_size = img_size
        self.Ncoil = Ncoil
        self.Ncontrast = Ncontrast
        self.Nsingular = par["Nsingular"]
        self.coilmap = coilmap
        self.window_size = window_size
        self.Nrep = Nrep
        self.kspaceMRF = kspaceMRF
        self.Nslice = Nslice
        self.npoints = npoints
        self.binnum = binnum
        self.view_dict_MRF = view_dict_MRF
        self.slice_dict_MRF = slice_dict_MRF
        self.inits = None