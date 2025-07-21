import torch
from torch import nn, optim
from torch import Tensor
import numpy as np
import numpy
import scipy.misc
import matplotlib.pyplot
import scipy.io as sio
import torchkbnufft as tkbn
from torch.fft import fftshift, fft, ifft
from torch.utils.checkpoint import checkpoint
from concurrent.futures import ThreadPoolExecutor

dtype = numpy.complex64


class Mypnufft_liverMRF_func_MRFMGRE_3D(torch.autograd.Function):

    @staticmethod
    def forward(ctx, singulars,  smap, ktraj, Ur_list_, slice_bin_list, dc, nufft_ob, adjnufft_ob):

        #coilmap : size(1 Ncoil 256   256  Nslice ), (x-dimension, y-dimension, coilmap dimension and real-image)
        #self.singulars: size(256, 256, Nslice, Nsingular, Necho, , 2)
        #dataMRF_bin shape: torch.Size([Npoint, Ncoil, N_readout, Nslice, Necho])
        #ktraj shape: torch.Size([N_readout, Npoint, 2])
        #Ur shape: torch.Size([Ncontrast, Nsingular])
        #dc: density compensition map (Npoint, 1)
        #Here N_readout is the number of readout lines a single motion state from
        #Which is determined by the window_size (which is the same for all ocntrasts)
        #the number of contrasts, the number of slice encoding in each contrast
        #N_readout = window_size(SUM_i(Nslice_i)) where i\in(0,Ncontrast)
        
        #input: Contr_slice_dict, size(N, 2), eg  Contr_slice_dict[0,:] = [40,72]
        #       ktraj: torch.Size([N, Npoint, 2])
        #Ur_list_: size(N, 4)
        #dc: density compensition map (Npoint, 1)
        dev = singulars.device
        smap = smap.unsqueeze(0).to(dev)
        Ncoil = smap.shape[1]
        #num_leaves_perC = ktraj.shape[1] # number of leaves per contrast 
        Npoint = dc.shape[0]
        Nslice = smap.shape[4]
        Necho = singulars.shape[4]
        Ur_list_ = Ur_list_.squeeze(0)
        Nsingular = Ur_list_.shape[1]
        img_size = singulars.shape[0]
        N_readout = Ur_list_.shape[0]
        #size(1, 256, 256, Nslice, Nsingular, Necho, 2)
        singulars = singulars.unsqueeze(0).permute(4,5,0,1,2,3,6) 
        #size(1, 256, 256, Nslice, Nsingular, Necho, 2)--> size(Nsingular, Necho, 1, 256, 256, Nslice 2)
        singulars = torch.view_as_complex(singulars).view(-1,1,img_size,img_size,Nslice) #size(Nsingular*Necho, 1, 256, 256, Nslice)
        singulars[:,:,:,:,1::2] = - singulars[:,:,:,:,1::2]
        singulars = singulars*smap
        singulars = fft(singulars,None,-1)
        #size(Nsingular, 1, 256, 256, Nslice)
        ktraj = ktraj.squeeze(0).float()
        #torch.Size([N, 288, 2])  
        
        def find_indices(lst, number):
            return [i for i, x in enumerate(lst) if x == number]

        #nufft_out_full = checkpoint(complex_func, Nslice,Npoint,Ncoil,N_readout,Necho,slice_bin_list,Ur_list_,singulars,ktraj,Nsingular,use_reentrant=False)
        nufft_out_full = torch.zeros(Npoint, Ncoil, N_readout, Necho).to(dtype = torch.complex64).to(dev)
        #print("running forward mynufft")
        for sli in range(Nslice):
            indices = find_indices(slice_bin_list.flatten(), sli)
            if len(indices)!=0:
                Ur_sli_list = Ur_list_[indices,:] #size(len(indices), Nsingular)
                out = singulars[:,:,:,:,sli].squeeze(-1)
                ktraj_slice = ktraj[indices,:,:].permute(2,1,0).contiguous().view(1,2,-1).expand(Nsingular*Necho,-1,-1) 
                #ktraj size: ( 2 288*len(indices))
                out = nufft_ob(out.to(dtype = torch.complex64), 
                                    ktraj_slice.to(dtype = torch.float32), 
                                    smaps = None)
                #torch.Size: (Nsingular*Necho 16 288*len(indices))
                out = out.view(Nsingular, Necho, Ncoil, Npoint, len(indices))
                out = out.permute(3,2,4,0,1) #torch.Size(288 16 len(indices) Nsingular Necho)
                out = torch.sum(out*Ur_sli_list.unsqueeze(0).unsqueeze(0).unsqueeze(-1),3).squeeze(-1) #torch.Size(288 16 len(indices))
                nufft_out_full[:,:,indices,:] = out.to(dtype = torch.complex64)

                del out
        ## ctx define
        ctx.Ncoil = Ncoil
        ctx.Npoint = Npoint
        ctx.Nslice = Nslice
        ctx.Nsingular = Nsingular
        ctx.Necho = Necho
        ctx.img_size = img_size
        ctx.Ur_list_ = Ur_list_
        ctx.smap = smap
        ctx.ktraj = ktraj
        ctx.dc = dc
        ctx.dev = dev
        ctx.slice_bin_list = slice_bin_list
        ctx.Ur_list_ = Ur_list_
        ctx.adjnufft_ob = adjnufft_ob
        

        return torch.view_as_real(nufft_out_full) #torch.Size(288 16 len(indices) Necho 2)

    @staticmethod
    def backward(ctx,grad_output):

        Ncoil = ctx.Ncoil
        # = ctx.num_leaves_perC 
        Nsingular = ctx.Nsingular
        Nslice = ctx.Nslice
        smap = ctx.smap 
        ktraj = ctx.ktraj
        dc = ctx.dc
        img_size = ctx.img_size
        slice_bin_list = ctx.slice_bin_list 
        Ur_list_ = ctx.Ur_list_
        dev = ctx.dev
        adjnufft_ob = ctx.adjnufft_ob
        Necho = ctx.Necho

        grad_output = torch.view_as_complex(grad_output) #torch.Size(Npoints 16 N_total Necho)
        grad_output = grad_output.permute(2,1,0,3) #torch.Size(N_total 16 Npoints Necho)
        grad_output = grad_output*dc.to(dev).reshape(1,1,-1,1)

        def find_indices(lst, number):
            return [i for i, x in enumerate(lst) if x == number]
        #slice_bin_list, N x 1, the list of slice indices in current bin
        #self.singulars: size(256, 256, Nsingular, Nslice, 2)
        out_full = torch.zeros(Nsingular*Necho, Ncoil,img_size,img_size,Nslice).to(dtype = torch.complex64).to(dev)
        for sli in range(Nslice):
            indices = find_indices(slice_bin_list.flatten(), sli)
            if len(indices)!=0:
                out = grad_output[indices,:,:,:].permute(3,1,2,0).contiguous().unsqueeze(0) #size(1, Necho, Ncoil, Npoint, len(indices))
                Ur_sli_list = Ur_list_[indices,:].permute(1,0).contiguous().unsqueeze(1).unsqueeze(2).unsqueeze(3) #size(Nsingular,1,1,1,len(indices))
                out = out*Ur_sli_list #size(Nsingular, Necho, Ncoil, Npoint, len(indices))
                out = out.view(Nsingular*Necho, Ncoil, -1)
                ktraj_slice = ktraj[indices,:,:].float().permute(2,1,0).contiguous().view(1,2,-1).expand(Nsingular*Necho,-1,-1)  #size(len(indices), 2, Npoint)
                #ktraj size: ( 2 288*len(indices))
                out = adjnufft_ob(out.to(dtype = torch.complex64), 
                                    ktraj_slice.to(dtype = torch.float32), 
                                    smaps = None)
                out_full[:,:,:,:,sli] = out.squeeze()
                
        out_full = ifft(fftshift(out_full,-1),None,-1)
        out_full = torch.sum(out_full*smap.conj(),1).squeeze()
        out_full = torch.view_as_real(out_full).permute(1,2,3,0,4).view(img_size,img_size, Nslice, Nsingular, Necho, 2) #Nsingular*Necho, 256,256, Nslice, 2  -->  256,256, Nslice, Nsingular*Necho, 2
        del out
    
        return out_full, None, None, None, None, None, None, None

 
 


class Mypnufft_liverMRFMGRE_3D_v2(nn.Module):
    def __init__(self, dc, coilmap):
        super(Mypnufft_liverMRFMGRE_3D_v2,self).__init__()
        self.dc = dc
        self.coilmap = coilmap
        dev = coilmap.device
        self.nufft_ob = tkbn.KbNufft(im_size=(256,256), grid_size = (320,320)).to(dev)
        self.adjnufft_ob = tkbn.KbNufftAdjoint(im_size=(256,256), grid_size = (320,320)).to(dev)
        
    def forward(self, singulars, ktraj, slice_bin_list, Ur_list):
        return  Mypnufft_liverMRF_func_MRFMGRE_3D.apply(singulars, self.coilmap, ktraj, Ur_list, slice_bin_list, self.dc, self.nufft_ob, self.adjnufft_ob)
