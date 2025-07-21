import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import scipy.io as sio
from encoding import get_encoder
from Mypnufft_mc_func_liver_MRF import Mypnufft_liverMRF_v2, Mypnufft_liverMRF_2D3D
from Mypnufft_mc_func_liver_MRF_3D import Mypnufft_liverMRF_3D_v2
from Mypnufft_mc_func_liver_MGRE_3D import Mypnufft_liverMRF_MGRE_3D
from Mypnufft_mc_func_liver_MRFMGRE_3D import Mypnufft_liverMRFMGRE_3D_v2, Mypnufft_liverMRFMGRE_3D_Toep
from math import sqrt
from models.unetVgg import unetVgg
from models.unet2D import Unet
from utils.transform_utils import fftn, ifftn



class SDFNetwork3D_STN_MRF_fat_water(nn.Module):
    def __init__(self,
                 dc,
                 coilmap,
                 par,
                 opt,
                 dfat,
                 te,
                 unet_dim = 0,
                 inits = None
                 ):
        super().__init__()
        self.Nslice = par["Nslice"]
        self.img_size = par["img_size"]
        self.Nsingular = par["Nsingular"] 
        self.unet_dim = unet_dim
        hidden_dim = opt.HashGrid_hidden_dim
        num_layers = opt.HashGrid_num_layers
        device = None
        gpu_ids = opt.gpu_ids
        self.device =  self.device = device if device is not None else torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')


        self.fw_net = SDFImageNet3D_allInOne(hidden_dim = hidden_dim, num_layers = num_layers, Nslice=self.Nslice,
                                       opt = opt, out_channel=self.Nsingular*2,num_groups=2)
        if unet_dim == 2:
            self.freq_net = Unet(input_channels=1,output_channels=1,num_filters=[2**i for i in range(3, 8)])
            self.r2s_net = Unet(input_channels=1,output_channels=1,num_filters=[2**i for i in range(3, 8)])
        elif unet_dim == 3:
            self.r2s_net = unetVgg(in_channels=1,out_channels=1)
            self.freq_net = unetVgg(in_channels=1,out_channels=1)
        elif unet_dim == 0:
            self.r2s_net = MagPhaseNet3D(par,num_layers=3,Nslice=self.Nslice,out_channel=1)
            self.freq_net = MagPhaseNet3D(par,num_layers=3,Nslice=self.Nslice,out_channel=1,agjust_mean_std = True)

        self.opt = opt

        self.mypnufft = Mypnufft_liverMRFMGRE_3D_v2(dc, coilmap)
        self.stn3D = STN3D(par)
        self.dfat = torch.tensor(dfat).to(self.device)
        self.dte = nn.Parameter(torch.zeros(1))
        self.te = torch.tensor(te)


    
    def forward(self, x, xt, ktraj, slice_bin_list, Ur_list, echo_indices = None, epoch = None):
        #x: x_image [B, 3]
        #xd: x_d [B, 4]
        dev = x.device
        dfat = self.dfat

        if echo_indices is not None:
            te = self.te.to(dev) 
            te = te[echo_indices]
        else: 
            te = self.te.to(dev) 
        Nsingular = self.Nsingular   
        fw, hd= self.fw_net(x,xt) #[1,img_size,img_size,Nslice,Nsingular*2, 2]
        #B0 = self.B0

        hi = fw.view(1,self.img_size, self.img_size,self.Nslice,-1).permute(0,4,1,2,3) #[1, Nsingular*2+1+3, img_size, img_size, Nslice
        h = self.stn3D(hi,hd) #[1, Nsingular*2+1+3,, img_size, img_size, Nslice]
        #h = hi
        h = h.permute(0,2,3,4,1).view(self.img_size, self.img_size,self.Nslice,-1).contiguous() #[img_size, img_size, Nslice, Nsingular*2+1+3]
        #h = hi.permute(0,2,3,4,1).view(self.img_size, self.img_size,self.Nslice,-1).contiguous()
        sw_r = h[...,0:Nsingular].unsqueeze(-1) #sinulgars_water_real
        sf_r = h[...,Nsingular:2*Nsingular].unsqueeze(-1) #sinulgars_fat_real
        sw_i = h[...,2*Nsingular:3*Nsingular].unsqueeze(-1) #sinulgars_water_imag
        sf_i = h[...,3*Nsingular:4*Nsingular].unsqueeze(-1) #sinulgars_fat_imag
        r2s = h[...,4*Nsingular].unsqueeze(-1).unsqueeze(-1)
        freq = h[...,-1].unsqueeze(-1).unsqueeze(-1)*100 #+ self.coef_mask
        

        te = te.view(1,1,1,1,-1)
        signal = ((sw_r + 1j*sw_i)  + (sf_r + 1j*sf_i) *  torch.exp(- 1j*2*math.pi * dfat * te)) * torch.exp(-r2s * 100* te)* torch.exp(-1j * 2 * math.pi * (freq)* te) #+ B0.unsqueeze(-1).unsqueeze(-1))* te) 
        if self.opt.istest:
            hi = hi.permute(0,2,3,4,1).squeeze().contiguous()
            static_sw_r = hi[...,0:Nsingular].unsqueeze(-1) #sinulgars_water_real; ...= [img_size, img_size, nslice]
            static_sf_r = hi[...,Nsingular:2*Nsingular].unsqueeze(-1)#sinulgars_fat_real; ...= [img_size, img_size, nslice]
            static_sw_i = hi[...,2*Nsingular:3*Nsingular].unsqueeze(-1)#sinulgars_water_imag; ...= [img_size, img_size, nslice]
            static_sf_i = hi[...,3*Nsingular:4*Nsingular].unsqueeze(-1) #sinulgars_fat_imag; ...= [img_size, img_size, nslice]
            r2s = hi[...,4*Nsingular].unsqueeze(-1).unsqueeze(-1)
            freq_static = hi[...,-1].unsqueeze(-1).unsqueeze(-1)
            #naming: s: singulars 
            static_water_s = (static_sw_r + 1j*static_sw_i)* torch.exp(-r2s * 100* te) * torch.exp(-1j * 2 * math.pi * freq * te) #size: [img_size, img_size, Nslice, Nsingulars, Necho]
            static_fat_s = (static_sf_r + 1j*static_sf_i)* torch.exp(-1j*2*math.pi * dfat * te)* torch.exp(-r2s * 100 * te) * torch.exp(-1j * 2 * math.pi * freq * 100 * te ) #size: [img_size, img_size, Nslice, Nsingulars, Necho]
            static_combined_s = ((sw_r + 1j*sw_i)  + (sf_r + 1j*sf_i) * torch.exp(- 1j*2*math.pi * dfat * te)) * torch.exp(-r2s * 100 * te) * torch.exp(-1j * 2 * math.pi * freq * 100 * te )
            static_data = { 
                'static_water_s': static_water_s, #size: [img_size, img_size, Nslice, Nsingulars, Necho]
                'static_fat_s': static_fat_s, 
                'static_combined_s': static_combined_s,
                'r2s' : r2s.squeeze()* 100,
                'freq' : freq_static.squeeze()*100,#+ B0,
            }
            dyna_water_s = sw_r + 1j*sw_i #size: [img_size, img_size, Nslice, Nsingulars]
            dyna_fat_s = (sf_r + 1j*sf_i) * torch.exp(- 1j*2*math.pi * dfat * te.squeeze()[0]) #size: [img_size, img_size, Nslice, Nsingulars]
            dyna_combined_s = (sw_r + 1j*sw_i)  + (sf_r + 1j*sf_i) * torch.exp(- 1j*2*math.pi * dfat * te.squeeze()[0])
            dyna_freq = freq
            dyna_data = {
                'dyna_water_s' : dyna_water_s.squeeze(),
                'dyna_fat_s' : dyna_fat_s.squeeze(),
                'dyna_combined_s': dyna_combined_s.squeeze(),
                'dyna_freq' : dyna_freq.squeeze()*100,#+ B0,
            }
            return 0, static_data, dyna_data, hd
        
        del sw_r, sf_r, sw_i, sf_i, r2s
        h = torch.view_as_real(signal)
        kdata = self.mypnufft(h, ktraj, slice_bin_list, Ur_list)
        return kdata, hi.permute(0,2,3,4,1).squeeze().contiguous(), hd, h, freq.squeeze()




    
class SDFImageNet3D_allInOne(nn.Module):
    def __init__(self,
                 Nslice,
                 opt,
                 encoding="hashgrid",
                 out_channel = 8,
                 num_layers=4,
                 skips=[],
                 hidden_dim=64,
                 clip_sdf=None,
                 num_groups = 1,
                 use_inits= False
                 ):
        super().__init__()
    
        self.num_layers = num_layers
        self.skips = skips
        self.hidden_dim = hidden_dim
        self.Nslice = Nslice      
        self.clip_sdf = clip_sdf
        num_levels = opt.HashGrid_num_lavels
        self.encoder, self.in_dim = get_encoder(encoding, num_levels=num_levels, input_dim=3,level_dim=2)
        self.encoder_QSM, self.in_dim_QSM = get_encoder(encoding, num_levels=num_levels, input_dim=3,level_dim=1)
        self.out_dim = out_channel//num_groups
        self.out_channel = out_channel
        self.num_groups = num_groups
        self.nets_real = []
        self.nets_imag = []
        self.opt = opt
        self.use_inits = use_inits
        
        if use_inits:
            in_dim = self.in_dim + 1
        else:
            in_dim = self.in_dim
 
        for l in range(num_groups):
            self.nets_real.append(MLP(in_dim, self.out_dim, self.num_layers, self.hidden_dim))
            self.nets_imag.append(MLP(in_dim, self.out_dim, self.num_layers, self.hidden_dim))
            
        self.net_r2s = MLP(in_dim, 1, self.num_layers, self.hidden_dim)
        self.net_freq = MLP(in_dim , 1, self.num_layers, self.hidden_dim)
        self.nets_real = nn.ModuleList(self.nets_real)
        self.nets_imag = nn.ModuleList(self.nets_imag)
        
        num_levels_D = opt.D_num_levels
        self.num_levels_D = num_levels_D
        num_layers_D = opt.D_num_layers
        hidden_dim_D = opt.D_hidden_dim

        self.encoder_D, self.in_dim_D = get_encoder(encoding,num_levels = num_levels_D, input_dim=4, level_dim=2)
        self.DisplaceNet = MLP(self.in_dim_D, 3, num_layers_D, hidden_dim_D)

        self.coef = nn.Parameter(torch.zeros(1,out_channel))
        self.coef_QSM = nn.Parameter(torch.zeros(1,1))

        
    def forward(self, x, xt, curlJ =None, init = None):
        # x: [B, 256*256, 2]
        # x: [1, 256*256, 2]
        x0 = x.squeeze() # x: [256*256, 2]
        x = self.encoder(x0)

        dev = x.device
        dtype = x.dtype
        dimxt = xt.shape[2]
        xt0 = xt.view(-1,dimxt)
        xt = self.encoder_D(xt0)

        if self.use_inits:
            mixed = init/torch.max(torch.abs(init))
            mixed.requires_grad = False
            h1 = torch.cat([x, torch.abs(mixed).to(dtype)], dim=-1)
            h2 = torch.cat([x, torch.abs(mixed).to(dtype)], dim=-1)
        else:
            h1 = x
            h2 = x
            h3 = x
            h4 = x #xfreq
            h5 = xt
            
        h1s = []
        h2s = []
        for i in range(len(self.nets_real)):
            h1s.append(self.nets_real[i](h1))
            h2s.append(self.nets_imag[i](h2))
        h1 = torch.cat(h1s,-1)*self.coef
        h2 = torch.cat(h2s,-1)*self.coef
        
        if self.r2s is not None:
            h3 = self.r2s.to(dev).to(dtype)
        else:
            h3 = self.net_r2s(h3)
        h4 = self.net_freq(h4)
        h5 = self.DisplaceNet(h5)
        
        img_size = int(sqrt(x.shape[0]/self.Nslice))
        h1 = h1.view(1,img_size,img_size,self.Nslice,-1)
        h2 = h2.view(1,img_size,img_size,self.Nslice,-1)
        h3 = h3.view(1,img_size,img_size,self.Nslice,-1)
        h4 = h4.view(1,img_size,img_size,self.Nslice,-1)
        h5 = h5.view(1,img_size,img_size,self.Nslice,-1)

        h = torch.cat([h1,h2,h3,h4],dim=-1)
        # need to reshape this h 
        #kdata = self.mypnufft(h, ktraj, smap)
        return h, h5


class MagPhaseNet3D(nn.Module):
    def __init__(self,
                 par,
                 Nslice,
                 encoding="hashgrid",
                 out_channel = 8,
                 num_layers=4,
                 level_dim = 2,
                 skips=[],
                 hidden_dim=64,
                 clip_sdf=None,
                 agjust_mean_std = False,
                 use_inits = False,
                 use_siren = False,
                 use_tanh = False
                 ):
        super().__init__()
        #can also be used for R2 map 
        self.num_layers = num_layers
        self.skips = skips
        self.hidden_dim = hidden_dim
        self.Nslice = Nslice
        self.clip_sdf = clip_sdf
        self.use_siren = use_siren
        self.encoder, self.in_dim = get_encoder(encoding,input_dim=3,level_dim=level_dim)
        self.mean = nn.Parameter(torch.zeros(out_channel))
        self.std = nn.Parameter(torch.zeros(out_channel))
        self.agjust_mean_std = agjust_mean_std
        self.use_inits = use_inits
        self.Nsingular = par["Nsingular"] 
        self.use_tanh = use_tanh
        backbone = []
        
        if use_siren:
            if use_inits:
                in_dim = self.in_dim + self.Nsingular
            else:
                in_dim = self.in_dim
            self.siren = Siren(in_dim, hidden_dim, num_layers, out_channel, outermost_linear= True)
        else:
            for l in range(num_layers):
                if l == 0:
                    if use_inits:
                        in_dim = self.in_dim + 1
                    else:
                        in_dim = self.in_dim
                elif l in self.skips:
                    in_dim = self.hidden_dim + self.in_dim
                else:
                    in_dim = self.hidden_dim
                
                if l == num_layers - 1:
                    out_dim = out_channel
                else:
                    out_dim = self.hidden_dim
                
                backbone.append(nn.Linear(in_dim, out_dim, bias=False))

            self.backbone = nn.ModuleList(backbone)
            self.tanh =  nn.Tanh()

    
    def forward(self, x, init=None):
        # x: [B, 256*256, 2]
        # x: [1, 256*256, 2]
        x = x.squeeze() # x: [256*256, 2]
        x = self.encoder(x)
        dtype = x.dtype
        if self.use_inits:
            init.requires_grad = False
            h1 = torch.cat([x, init.to(dtype)], dim=-1)
        else:
            h1 = x
            
        if self.use_siren:
            h1 = self.siren(h1)
        else:
            for l in range(self.num_layers):
                if l in self.skips:
                    h1 = torch.cat([h1, x], dim=-1)
                h1 = self.backbone[l](h1)
                if l != self.num_layers - 1:
                    h1 = F.relu(h1, inplace=True)
            if self.clip_sdf is not None:
                h1 = h1.clamp(-self.clip_sdf, self.clip_sdf)
            if self.use_tanh:
                h1 = self.tanh(h1)
        
        img_size = int(sqrt(x.shape[0]/self.Nslice))
        Nch = int(h1.shape[1])
        h1 = h1.view(1,img_size,img_size,self.Nslice,Nch)
        if self.agjust_mean_std:
            h1 = h1*self.std.view(1,1,1,1,-1) + self.mean.view(1,1,1,1,-1)
        # need to reshape this h 
        #kdata = self.mypnufft(h, ktraj, smap)
        return h1




class MLP(nn.Module):
    def __init__(self,
                in_channel = 32,
                out_channel = 8,
                num_layers=4,
                hidden_dim=64,
                use_tanh = False,
                clamp = None,
                dropout_rate = 0.0,
                 ):
        super().__init__()


        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_tanh = use_tanh
        backbone = []
        for l in range(num_layers):
            if l == 0:
                in_dim = in_channel
            else:
                in_dim = self.hidden_dim
            
            if l == num_layers - 1:
                out_dim = out_channel
            else:
                out_dim = self.hidden_dim
            backbone.append(nn.Linear(in_dim, out_dim, bias=False))
        self.backbone = nn.ModuleList(backbone)
        self.tanh = nn.Tanh()
        self.clamp = clamp
        self.ELU = torch.nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self,h):
        # x: [B, 256*256, 2]
        # x: [1, 256*256, 2]
        for l in range(self.num_layers):
            h = self.backbone[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)
                #h = self.ELU(h)
                h = self.dropout(h)
        if self.clamp is not None:
            h = h.clamp(-self.clamp, self.clamp)
        if self.use_tanh:
            h = self.tanh(h)
        return h
    
    
      
class STN3D(nn.Module):
    #spatial transformer network
    def __init__(self,par):
        super().__init__()
        #cartetian 3d locations
        self.Nslice = par["Nslice"]
        self.img_size = par["img_size"]
        self.width = self.img_size
        self.height = self.img_size
        self.depth = self.Nslice
        # modified from (depth, height, width)
        
    def forward(self, img, D):
        #print("in.shape: "+str(img.shape))
        #print("D shape:" +str(D.shape))
        #print("cart_3d_locations shape:" +str(self.cart_3d_locations.shape))
        dev = D.device
        sub_batch_size = D.shape[0]
        new_grid = self.get_3d_locations(self.width, self.height, self.depth).expand(sub_batch_size,-1,-1,-1,-1).to(dev) + D
        #net_output: size(batch_size, 8, img_size, img_size, Nslice)
        out = F.grid_sample(input=img, grid=new_grid, mode='bilinear', padding_mode="border",  align_corners=False)
        #print("out.shape: "+str(out.shape))
        #out: size(batch_size, 8, img_size, img_size, Nslice)
        return out
    
    def get_3d_locations(self, d,h,w):
        locations_x = torch.linspace(0, w-1, w).view(1, 1, 1, w).expand(1, d, h, w)
        locations_y = torch.linspace(0, h-1, h).view(1, 1, h, 1).expand(1, d, h, w)
        locations_z = torch.linspace(0, d-1,d).view(1, d, 1, 1).expand(1, d, h, w)
        locations_x = (2.0*locations_x - (w-1))/(w-1)
        locations_y = (2.0*locations_y - (h-1))/(h-1)
        locations_z = (2.0*locations_z - (d-1))/(d-1)
        # stack locations
        locations_3d = torch.stack([locations_x, locations_y, locations_z], dim=4)
        return locations_3d
    
      
 