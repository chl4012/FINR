import os
import glob
import tqdm
import random
import warnings
import tensorboardX
from utils.image_utils import grad, gradient_mask, erode_mask, sphere_kernel
import numpy as np
from unwrap import unwrap

import time
from datetime import datetime

import h5py
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import mat73

from rich.console import Console
from torch_ema import ExponentialMovingAverage
import socket
from contextlib import closing
import packaging
from collections import OrderedDict

from utils.fits import arlo_numpy, arlo, fit_complex

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if packaging.version.parse(torch.__version__) < packaging.version.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


def get_open_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class Trainer(object):
    def __init__(self, 
                 name, # name of this experiment
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 use_multi_gpus=False, # use multiple GPUs?
                 gpu_ids = [0], # gpu ids for parallel computing
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=10, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metirc
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 par = None,
                 opt = None
                 ):
        
        self.par = par
        self.opt = opt
        self.name = name
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.use_multi_gpus = use_multi_gpus
        self.gpu_ids = gpu_ids
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        if use_multi_gpus:
            master_gpu = gpu_ids[0]
            self.device = device if device is not None else torch.device(f'cuda:{master_gpu}' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device if device is not None else torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.count = 1
        self.count_QSM = 1

        model.to(self.device)
        if self.use_multi_gpus:
            #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
            model = model.cuda(gpu_ids[0])
            #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion
        
        if self.opt.Dnet_ckp is not None:
            self.load_Dnet_checkpoint(self.opt.Dnet_ckp)
            
        if optimizer is None:
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.opt.lr, betas=(0.9, 0.99), eps=1e-15) # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        print("workspace: " + self.workspace)
        if self.workspace is not None:
            
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")
            self.ckpt_path = os.path.join(self.workspace, 'ckp')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth.tar"
            self.saved_data_path = os.path.join(self.workspace, 'data')
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
        

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)
                


    def __del__(self):
        """
        if self.log_ptr: 
            self.log_ptr.close()
        """

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    ### ------------------------------	

    def train_step(self, data):

        # assert batch_size == 1

        if self.name == "MRF_fat_water_STN":
            X = data["points_x"] # [B, 3]
            X_t = data["points_xt"]
            ktraj = data["ktraj"]
            slice_bin_list = data["slice_bin_list"] 
            y = data["dataMRF_bin"] # [B]
            Ur_list = data["Ur_bin_list"]
            Necho = self.par["Necho"]
            
            if self.epoch <2:
                echo_indices = np.array([0, 1, 2])
            else:
                echo_indices = np.random.choice(range(1,Necho), size=2, replace=False)
                echo_indices = np.append(echo_indices, 0)
            
            y = y[:,:,:,:,echo_indices,:]
            
            if self.opt:
                if self.opt.istest:
                    if self.opt.use_direct:
                        pred, static_data = self.model(X, X_t, ktraj, slice_bin_list, Ur_list)
                        loss = self.criterion(torch.zeros(1), torch.zeros(1))
                        return loss, static_data
                    else:
                        pred, static_data, dyna_data, Dfield = self.model(X, X_t, ktraj, slice_bin_list, Ur_list)
                        loss = self.criterion(torch.zeros(1), torch.zeros(1))
                        return loss, static_data, dyna_data, Dfield

            pred, singulars, Dfield, movingSingulars, mask = self.model(X, X_t, ktraj, slice_bin_list, Ur_list, echo_indices, epoch = self.epoch) #kdata, (self.img_size,self.img_size,self.Nslice,Nsingular*2+1+3), hd, h
            motion_diff = data["map_in"] #the difference between the current motion value and the end of inspiration
            # maps 0 to 100, maps 1 to 1
            def exp_decay(x):
                a = 100
                b = 1 / 100
                return a * (b ** x)    
            loss1 = self.criterion(pred, y)
            loss2 = (0.1)*gradL2Reg4D(Dfield)
            Nsingular = self.par["Nsingular"]
            loss = loss1 #+ loss2

            return pred, y, loss, singulars, Dfield, movingSingulars 

        
        
        return pred, y, loss, singulars

    def eval_step(self, data):
        return self.train_step(data)

    def test_step(self, data):  
        X = data["points"][0]
        pred = self.model(X)
        return pred        

    ### ------------------------------

    def train(self, train_loader, max_epochs , train_Loader_flat = None):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))
        
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            if (train_Loader_flat is not None) and epoch<3:
                print("training with flat data")
                self.train_one_epoch(train_Loader_flat) #Chao: 07/14/2024
            else:
                self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()


    def evaluate(self, loader):
        #if os.path.exists(self.best_path):
        #    self.load_checkpoint(self.best_path)
        #else:
        #    self.load_checkpoint()
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader)
        self.use_tensorboardX = use_tensorboardX



    def prepare_data(self, data):
        if isinstance(data, list):
            for i, v in enumerate(data):
                if isinstance(v, np.ndarray):
                    data[i] = torch.from_numpy(v).to(self.device, non_blocking=True)
                if torch.is_tensor(v):
                    data[i] = v.to(self.device, non_blocking=True)
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    data[k] = torch.from_numpy(v).to(self.device, non_blocking=True)
                if torch.is_tensor(v):
                    data[k] = v.to(self.device, non_blocking=True)
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(self.device, non_blocking=True)
        else: # is_tensor, or other similar objects that has `to`
            data = data.to(self.device, non_blocking=True)

        return data


    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        """
        if self.use_multi_gpus:
            loader.sampler.set_epoch(self.epoch)
        """
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for idx, data in enumerate(loader):
            
            self.local_step += 1
            self.global_step += 1
            
            data = self.prepare_data(data)

            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=self.fp16):
                if self.name == "HashGrid3D_STN"  \
                    or self.name == "HashGrid3DMGRE_STN"  \
                    or self.name == "MRF_fat_water_STN" \
                    or self.name == "pure_fat_water_MGRE"\
                    or self.name == "MRF_fat_water_2_route":
                    preds, truths, loss, singulars, Dfield, _ = self.train_step(data)
                elif self.name == "MRF_fat_water_STN_QSM":
                    preds, truths, loss, singulars, Dfield, _, freq_bg, freq_local = self.train_step(data)
                elif self.name == "pretrain_dictionary":
                    preds, truths, loss = self.train_step(data)
                elif self.name == "brain_mcTFI":
                    preds, truths, loss, QSM, r2s, magnitude, error_map = self.train_step(data)
                    singulars = 0
                    Dfield = 0
                    _ = 0
                elif self.name == 'FGN':
                    preds, truths, loss = self.train_step(data)
                elif self.name == 'PEN':
                    preds, truths, T1, T2, M0, loss = self.train_step(data)
                else:
                    preds, truths, loss, singulars = self.train_step(data)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.epoch:

                if self.name == "MRF_fat_water_STN":
                    self.save_images_water_fat(singulars, Dfield, idx)
                else:
                    motion_value = data['map_in'][0]
                    self.save_images(singulars, idx)
                

            if self.ema is not None:
                self.ema.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)
            
            if self.name == "FGN":
                del preds, truths, loss
            elif self.name == "PEN":
                del preds, truths, T1, T2, M0, loss
            else:
                del preds, truths, loss, singulars, Dfield, _

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def save_images(self, out_net, idx):
        if self.name == "HashGrid2D":
            outnet_2 = torch.view_as_complex(out_net)                
            outnet_2_save = torch.abs(outnet_2) #(batch_size, 256, 256,4)
            outnet_2_save_2 = torch.zeros((4,256,256))
            for i in range(4):                             
                outnet_2_save_2[i,:,:] = outnet_2_save[0,:,:,i].squeeze()/torch.max(outnet_2_save[0,:,:,i])
            plt.imsave(self.ckpt_path+'/epoch_{}.png'.format(self.epoch), outnet_2_save_2.reshape(256*4,256).detach().cpu().numpy(), cmap="gray")
        elif self.name == "HashGrid3D" or self.name == "HashGrid3D_STN":
            Nslice = self.par['Nslice']
            Nsingular = self.par["Nsingular"]
            outnet_2 = torch.view_as_complex(out_net)  
            outnet_2_save = torch.abs(outnet_2).squeeze() #(256, 256, Nslice, 4, 2)
            outnet_2_save = outnet_2_save[:,:,Nslice//2,:].squeeze()
            outnet_2_save_2 = torch.zeros((Nsingular,256,256))
            for i in range(Nsingular):                            
                outnet_2_save_2[i,:,:] = outnet_2_save[:,:,i].squeeze()/torch.max(outnet_2_save[:,:,i])
            plt.imsave(self.ckpt_path+'/epoch_{}_batch_{}.png'.format(self.epoch,idx), outnet_2_save_2.reshape(256*Nsingular,256).detach().cpu().numpy(), cmap="gray")
        elif self.name == "HashDfield4D":
            outnet_2_save_ = torch.abs(out_net) #(256, 256, Nslice, 3)
            batch_size = outnet_2_save_.shape[0]
            print(batch_size)
            for b in range(batch_size):
                outnet_2_save = outnet_2_save_[b]
                outnet_2_save = outnet_2_save[:,:,35,:].squeeze()
                nch = outnet_2_save.shape[-1]
                outnet_2_save_2 = torch.zeros((nch,256,256))
                for i in range(nch):                            
                    outnet_2_save_2[i,:,:] = outnet_2_save[:,:,i].squeeze()/torch.max(outnet_2_save[:,:,i])
                plt.imsave(self.ckpt_path+'/epoch_{}_batch_{}_idx_{}.png'.format(self.epoch,idx,b), outnet_2_save_2.reshape(256*nch,256).detach().cpu().numpy(), cmap="gray")
        else:
            outnet_2 = torch.view_as_complex(out_net)                
            outnet_2_save = torch.abs(outnet_2).squeeze() #(batch_size, 256, 256,4)
            outnet_2_save_2 = torch.zeros((4,256,256))
            for i in range(4):                            
                outnet_2_save_2[i,:,:] = outnet_2_save[0,:,:,i].squeeze()/torch.max(outnet_2_save[0,:,:,i])
            plt.imsave(self.ckpt_path+'/epoch_{}.png'.format(self.epoch), outnet_2_save_2.reshape(256*4,256).detach().cpu().numpy(), cmap="gray")

    
    def save_images_water_fat(self, out_net, Dfield, idx):
        Nslice = self.par['Nslice']
        Nsingular = self.par["Nsingular"]
        #out_net: [img_size, img_size, Nslice, Nsingular*2+1+3]
        slices = [14,17,24]
        slice = 24
        for slice in slices:
            sw_r = out_net[:,:,slice,0:Nsingular].detach().cpu() #sinulgars_water_real
            sf_r = out_net[:,:,slice,Nsingular:2*Nsingular].detach().cpu() #sinulgars_fat_real
            sw_i = out_net[:,:,slice,2*Nsingular:3*Nsingular].detach().cpu()#sinulgars_water_imag
            sf_i = out_net[:,:,slice,3*Nsingular:4*Nsingular].detach().cpu() #sinulgars_fat_imag
            r2s = out_net[:,:,slice,4*Nsingular].detach().cpu()
            freq = out_net[:,:,slice,-1].detach().cpu()
            
            outnet_2_save_2 = torch.zeros((Nsingular,256,256))
            s_mw = torch.abs(sw_r + 1j*sw_i)
            for i in range(Nsingular):                            
                outnet_2_save_2[i,:,:] = s_mw[:,:,i].squeeze()/torch.max(s_mw[:,:,i])
            plt.imsave(self.ckpt_path+'/epoch_{}_batch_{}_ws_slice_{}.png'.format(self.epoch,idx,slice), outnet_2_save_2.reshape(256*Nsingular,256).numpy(), cmap="gray")

            outnet_2_save_2 = torch.zeros((Nsingular,256,256))
            s_mf = torch.abs(sf_r + 1j*sf_i)
            for i in range(Nsingular):                            
                outnet_2_save_2[i,:,:] = s_mf[:,:,i].squeeze()/torch.max(s_mf[:,:,i])
            plt.imsave(self.ckpt_path+'/epoch_{}_batch_{}_fs_slice_{}.png'.format(self.epoch,idx,slice), outnet_2_save_2.reshape(256*Nsingular,256).numpy(), cmap="gray")
        
            #w_ph = w_ph/torch.max(w_ph)
            #f_ph = f_ph/torch.max(f_ph)
            freq = freq/torch.max(freq)
            #phases = torch.cat((w_ph,f_ph,freq),dim = 0)
            phases = freq
            plt.imsave(self.ckpt_path+'/epoch_{}_batch_{}_phases_slice_{}.png'.format(self.epoch,idx,slice), phases.numpy(), cmap="gray")
            
            r2s = r2s/torch.max(r2s)
            plt.imsave(self.ckpt_path+'/epoch_{}_batch_{}_r2s_slice_{}.png'.format(self.epoch,idx,slice), r2s.numpy(), cmap="gray")

        outnet_3 = Dfield.squeeze()
        outnet_3_save = outnet_3.squeeze() #(256, 256, Nslice, 4, 2)
        outnet_3_save = outnet_3_save[:,:,Nslice//2,:].squeeze()
        outnet_3_save_2 = torch.zeros((3,256,256))
        for i in range(3):                            
            outnet_3_save_2[i,:,:] = outnet_3_save[:,:,i].squeeze()/torch.max(torch.abs(outnet_3_save[:,:,i]))
        plt.imsave(self.ckpt_path+'/epoch_{}_batch_{}_D.png'.format(self.epoch,idx), outnet_3_save_2.reshape(256*3,256).detach().cpu().numpy(), cmap="gray")

        
    def evaluate_one_epoch(self, loader):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        save_path = os.path.join(self.workspace, 'test')
        os.makedirs(save_path,exist_ok=True)
        save_path_sub = os.path.join(save_path, f'{self.name}_{self.epoch}.mat')
        

        if self.name == 'MRF_fat_water_STN':
            Nslice = self.par['Nslice']
            Necho = self.par["Necho"]
            Nsingular = self.par["Nsingular"]
            #am: all motion; ae: all echoes; as: all singulars
            Dfield_am =  np.zeros((256,256,Nslice,3,len(loader)),dtype = complex)
            dyn_water_as_am =  np.zeros((256,256,Nslice,Nsingular,len(loader)),dtype = complex)
            dyn_fat_as_am =  np.zeros((256,256,Nslice,Nsingular,len(loader)),dtype = complex)
            dyn_combined_as_am =  np.zeros((256,256,Nslice,Nsingular,len(loader)),dtype = complex)
            BGQSM =  np.zeros((256,256,Nslice,len(loader)),dtype = complex)
            dyna_freq = np.zeros((256,256,Nslice,len(loader)),dtype = complex)
        else:
            save_img =  np.zeros((256,256,4,len(loader)),dtype = complex)
        
        
        with torch.no_grad():
            self.local_step = 0
            for _, data in enumerate(loader):    
                self.local_step += 1
                
                data = self.prepare_data(data)

                if self.ema is not None:
                    self.ema.store()
                    self.ema.copy_to()
            
                with torch.cuda.amp.autocast(enabled=self.fp16):

                    if  self.name == 'MRF_fat_water_STN':
                        loss, static_data, dyna_data, Dfield = self.train_step(data)
                    else:
                        preds, truths, loss, singulars = self.eval_step(data)
                
                idx = int(data['idx_bin'])
                if self.name == 'HashGrid2D3D':
                    idx = int(data['slice'])
                

                if self.name == 'MRF_fat_water_STN' :
                    if self.opt.use_direct:
                        save_path_sub = os.path.join(save_path, 'singular_echo.mat')
                        sio.savemat(save_path_sub,{"singular_echo":(torch.real(static_data).float()+1j*torch.imag(static_data).float()).detach().cpu().numpy()})
                        print("Static images saved; Process finished")
                        return
                    else:
                        if idx == 0:
                            static_water_as_ae = static_data['static_water_s'].detach().cpu().numpy()
                            static_fat_as_ae = static_data['static_fat_s'].detach().cpu().numpy()
                            static_combined_as_ae = static_data['static_combined_s'].detach().cpu().numpy()
                            r2s = static_data['r2s'].detach().cpu().numpy()
                            freq = static_data['freq'].detach().cpu().numpy()
                        Dfield_am[:,:,:,:,idx] = Dfield.squeeze(0).cpu().numpy()
                        dyn_water_as_am[:,:,:,:,idx] = dyna_data['dyna_water_s'].detach().cpu().numpy()
                        dyn_fat_as_am[:,:,:,:,idx] = dyna_data['dyna_fat_s'].detach().cpu().numpy()
                        dyn_combined_as_am[:,:,:,:,idx] = dyna_data['dyna_combined_s'].detach().cpu().numpy()   

                else:
                    save_img[:,:,:,idx] = torch.view_as_complex(singulars.float().squeeze(0)).cpu().numpy()


                if self.ema is not None:
                    self.ema.restore()
                
                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.use_multi_gpus:
                    #dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / len(self.gpu_ids)
                    

                loss_val = loss.item()
                total_loss += loss_val
                if self.name == "HashDfield4D":
                    save_path_sub = os.path.join(save_path, f'{self.name}_{idx}.mat')
                    sio.savemat(save_path_sub,{f'{self.name}_{idx}':singulars})

                if self.local_rank == 0:
                    for metric in self.metrics:
                        if not self.opt.istest:
                            metric.update(preds, truths)
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)
                    

            if self.name == 'MRF_fat_water_STN':  
                def save_h5(save_path_root, var_name, data):
                    save_path_sub = os.path.join(save_path_root, var_name+'.h5')
                    with h5py.File(save_path_sub, 'w') as hdf:
                        hdf.create_dataset(var_name, data=data)
                Ur = self.par['Ur'] 
                te = self.par['te']
                dfat = self.par['dfat']
                print("------saving "+'static_water_as_ae-----')
                save_h5(save_path, 'static_water_as_ae_real', static_water_as_ae.real)
                save_h5(save_path, 'static_water_as_ae_imag', static_water_as_ae.imag)
                print("------saving "+'static_fat_as_ae-----')
                save_h5(save_path, 'static_fat_as_ae_real', static_fat_as_ae.real)
                save_h5(save_path, 'static_fat_as_ae_imag', static_fat_as_ae.imag)
                print("------saving "+'static_combined_as_ae-----')
                save_h5(save_path, 'static_combined_as_ae_real', static_combined_as_ae.real)
                save_h5(save_path, 'static_combined_as_ae_imag', static_combined_as_ae.imag)
                print("------saving "+'r2s-----')
                save_h5(save_path, 'r2s', r2s)
                print("------saving "+'freq-----')
                save_h5(save_path, 'freq', freq)
                print("------saving "+'BGQSM-----')

                print("------saving "+'dynamic freq-----')
                save_h5(save_path, 'dyn_freq', dyna_freq.real)
                print("------saving "+'dyn_water_as_am-----')
                save_h5(save_path, 'dyn_water_as_am_real', dyn_water_as_am.real)
                save_h5(save_path, 'dyn_water_as_am_imag', dyn_water_as_am.imag)
                print("------saving "+'dyn_fat_as_am-----')
                save_h5(save_path, 'dyn_fat_as_am_real', dyn_fat_as_am.real)
                save_h5(save_path, 'dyn_fat_as_am_imag', dyn_fat_as_am.imag)
                print("------saving "+'dyn_combined_as_am-----')
                save_h5(save_path, 'dyn_combined_as_am_real', dyn_combined_as_am.real)
                save_h5(save_path, 'dyn_combined_as_am_imag', dyn_combined_as_am.imag)
                print("------saving "+'Ur te dfat-----')
                save_h5(save_path, 'Ur', Ur)
                save_h5(save_path, 'te', te)
                save_h5(save_path, 'dfat', dfat)
                print("------saving "+'Dfield-----')
                save_h5(save_path, 'Dfield', Dfield_am.real)
            
                

                
        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss
            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, full=False, best=False):

        state = {
            'epoch': self.epoch,
            'stats': self.stats,
        }

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{self.name}_ep{self.epoch:04d}.pth.tar"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = self.stats["checkpoints"].pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)
            
            torch.save(state, file_path)

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None):

        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth.tar'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        #print(checkpoint_dict['model'])
        #checkpoint_dict = self.convert_state_dict(checkpoint_dict)
        #print(checkpoint_dict.keys())
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(self.convert_state_dict(checkpoint_dict['model']), strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")        

        #if self.ema is not None and 'ema' in checkpoint_dict:
        #   self.ema.load_state_dict(checkpoint_dict['ema'])

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        
        if self.optimizer and  'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer, use default.")
        
        # strange bug: keyerror 'lr_lambdas'
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler, use default.")

        if 'scaler' in checkpoint_dict:
            self.scaler.load_state_dict(checkpoint_dict['scaler'])



    def load_FGN_checkpoint_for_PEN(self, FGN_model, checkpoint=None):
        self.log("loading FGN parameters and fixing these parameters ...")
        print("checkpoint path: ", self.ckpt_path)
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/FGN_ep*.pth.tar'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            FGN_model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return
        
        def convert_state_dict(state_dict):
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v  # remove 'module.' prefix
                else:
                    new_state_dict[k] = v
            return new_state_dict
                    
        missing_keys, unexpected_keys = FGN_model.load_state_dict(convert_state_dict(checkpoint_dict['model']), strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")        

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        
        if self.optimizer and  'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer, use default.")
        
        # strange bug: keyerror 'lr_lambdas'
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler, use default.")

        if 'scaler' in checkpoint_dict:
            self.scaler.load_state_dict(checkpoint_dict['scaler'])
            
        self.log("[INFO] Fixing FGN model parameters to non-trainable...")
        for param in FGN_model.parameters():
            param.requires_grad = False
            
            
            
    def load_Dnet_checkpoint(self, checkpoint):
        print('loading displaceNet parameters and fixing these prameters ...')
        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        model_state_dict = checkpoint_dict['model']
        first_key = next(iter(model_state_dict.keys()))
        if first_key.startswith('module'):
            prefix_encoder = 'module.all_net.encoder_D.'
            prefix_MLP = 'module.all_net.DisplaceNet.'
        else:
            prefix_encoder = 'all_net.encoder_D.'
            prefix_MLP = 'all_net.DisplaceNet.'
        encoder_D_state_dict = {}
        MLP_state_dict = {}
        
          # Prefix to remove from the loaded state dict keys
        for k, v in model_state_dict.items():
            # Remove prefix and reassign the value to the new key
            if k.startswith(prefix_encoder):
                new_key = k[len(prefix_encoder):] 
                encoder_D_state_dict[new_key] = v
                print(k)
                print(v)
            else:
                continue
            
        for k, v in model_state_dict.items():
            # Remove prefix and reassign the value to the new key
            if k.startswith(prefix_MLP):
                new_key = k[len(prefix_MLP):] 
                MLP_state_dict[new_key] = v
            else:
                continue
        # Load the modified state dict into your model

        missing_keys, unexpected_keys = self.model.module.fw_net.encoder_D.load_state_dict(encoder_D_state_dict)
        missing_keys, unexpected_keys = self.model.module.fw_net.DisplaceNet.load_state_dict(MLP_state_dict)
        for param in self.model.module.fw_net.encoder_D.parameters():
            param.requires_grad = False
        for param in self.model.module.fw_net.DisplaceNet.parameters():
            param.requires_grad = False


    
    def load_FGN_checkpoint(self, checkpoint):
        def get_checkpoint(self):
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/FGN_ep*.pth.tar'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint for FGN is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found for FGN, model randomly initialized.")
                return checkpoint
            
        print('loading the FGN parameters and fixing these prameters ...')
        checkpoint = get_checkpoint(self)
        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        model_state_dict = checkpoint_dict['model']
        first_key = next(iter(model_state_dict.keys()))
        if first_key.startswith('module'):
            prefix_MLP = 'module.MLP.'
        else:
            prefix_MLP = 'MLP.'
        MLP_state_dict = {}
        # Prefix to remove from the loaded state dict keys
        for k, v in model_state_dict.items():
            # Remove prefix and reassign the value to the new key
            if k.startswith(prefix_MLP):
                new_key = k[len(prefix_MLP):] 
                MLP_state_dict[new_key] = v
            else:
                continue
        # Load the modified state dict into your model
        missing_keys, unexpected_keys = self.model.module.fw_net.DisplaceNet.load_state_dict(MLP_state_dict)
        self.log("[INFO] loaded FGN model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")      
        for param in self.model.module.FGN.MLP.parameters():
            param.requires_grad = False


            
    def convert_state_dict(self, state_dict):
        
        new_state_dict = OrderedDict()

        if self.use_multi_gpus:
            if next(iter(state_dict)).startswith("module."):
                return state_dict  # abort if dict is a DataParallel model_state

            for k, v in state_dict.items():
                name = 'module.' + k  # add `module.`
                new_state_dict[name] = v
        else:

            if not next(iter(state_dict)).startswith("module."):
                return state_dict  # abort if dict is not a DataParallel model_state

            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v

        return new_state_dict
    
    def gradL2JTV4D(self, field):
        field = field.squeeze(0)
        # field: (1,img_size,img_size,self.Nslice,3)
        Nsingular = self.par["Nsingular"]
        sw_r = field[...,0:Nsingular].detach().cpu() #sinulgars_water_real
        sw_i = field[...,2*Nsingular:3*Nsingular].detach().cpu()#sinulgars_water_imag

        sw_abs = torch.abs(sw_r + 1j*sw_i).view(-1,Nsingular)
        sw_abs_mean = torch.mean(sw_abs,0).view(1,1,1,Nsingular)
        sw_abs = sw_abs/sw_abs_mean   
        
        Dx = torch.cat((sw_abs[1:,:,:,:], sw_abs[-1:,:,:,:]),0) - sw_abs
        Dy = torch.cat((sw_abs[:,1:,:,:], sw_abs[:,-1:,:,:]),1) - sw_abs
        Dz = torch.cat((sw_abs[:,:,1:,:], sw_abs[:,:,-1:,:]),2) - sw_abs
        
        D_square = torch.sum(torch.square(Dx) + torch.square(Dy) + torch.square(Dz), -1)
        reg = torch.sqrt(D_square).sum()
        
        return reg

def gradL2Reg4D(field):
    field = field.squeeze(0)
    # field: (1,img_size,img_size,self.Nslice,3)
    Dx = torch.cat((field[1:,:,:,:], field[-1:,:,:,:]),0) - field
    Dy = torch.cat((field[:,1:,:,:], field[:,-1:,:,:]),1) - field
    Dz = torch.cat((field[:,:,1:,:], field[:,:,-1:,:]),2) - field
    reg = torch.square(Dx).mean() + torch.square(Dy).mean() + torch.square(Dz).mean()
    return reg

def gradL1Reg4D(field):
    field = field.squeeze(0)
    # field: (1,img_size,img_size,self.Nslice,3)
    Dx = torch.cat((field[1:,:,:,:], field[-1:,:,:,:]),0) - field
    Dy = torch.cat((field[:,1:,:,:], field[:,-1:,:,:]),1) - field
    Dz = torch.cat((field[:,:,1:,:], field[:,:,-1:,:]),2) - field
    reg = torch.abs(Dx).sum() + torch.abs(Dy).sum() + torch.abs(Dz).sum()
    return reg



def gradL1Reg3D(field):
    field = field.squeeze(0)
    # field: (1,img_size,img_size,self.Nslice,3)
    Dx = torch.cat((field[1:,:,:], field[-1:,:,:]),0) - field
    Dy = torch.cat((field[:,1:,:], field[:,-1,:]),1) - field
    Dz = torch.cat((field[:,:,1:], field[:,:,-1]),2) - field
    reg = torch.abs(Dx).sum() + torch.abs(Dy).sum() + torch.abs(Dz).sum()
    return reg

def gradL2Reg5D(field):
    field = field.squeeze(0)
    # field: (1,img_size,img_size,self.Nslice,3)
    Dx = torch.cat((field[1:,:,:,:,:], field[-1:,:,:,:,:]),0) - field
    Dy = torch.cat((field[:,1:,:,:,:], field[:,-1:,:,:,:]),1) - field
    Dz = torch.cat((field[:,:,1:,:,:], field[:,:,-1:,:,:]),2) - field
    reg = torch.square(Dx).sum() + torch.square(Dy).sum() + 2*torch.square(Dz).sum()
    return reg

def gradL2Reg5D_t(field):
    Dt = torch.cat((field[1:,:,:,:,:], field[-1:,:,:,:,:]),0) - field
    reg = torch.mean(torch.square(Dt))
    return reg

def gradL1Reg5D(field):
    field = field.squeeze()
    # field: (1,img_size,img_size,self.Nslice,3)
    Dx = torch.cat((field[1:,:,:,:,:], field[-1:,:,:,:,:]),0) - field
    Dy = torch.cat((field[:,1:,:,:,:], field[:,-1:,:,:,:]),1) - field
    Dz = torch.cat((field[:,:,1:,:,:], field[:,:,-1:,:,:]),2) - field
    reg = torch.abs(Dx).sum() + torch.abs(Dy).sum() + torch.abs(Dz).sum()
    return reg