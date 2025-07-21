import torch
import torch.nn as nn
import torchvision
import sys

import numpy as np
from PIL import Image
import PIL
import numpy as np

import matplotlib.pyplot as plt
from math import pi


def get_params(opt_over, net, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
    
        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif opt == 'all':
            net_input.requires_grad = True
            params += [net_input]
            params += [x for x in net.parameters() ]
        else:
            assert False, 'what is it?'
            
    return params




def get_R2s_joint_params(opt_over, net, r2s_net, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
    
        if opt == 'net':
            params += [x for x in net.parameters() ]
            params += [x for x in r2s_net.parameters() ]
        elif opt == 'all':
            net_input.requires_grad = True
            params += [net_input]
            params += [x for x in net.parameters() ]
        else:
            assert False, 'what is it?'
            
    return params

def get_input_manifold(input_type, motion_indices, times):
    if input_type.startswith('2Dline'):
        #motion_indices size([1,1600])
        #times size([1,1600])
        
        manifold = torch.tensor(np.concatenate((motion_indices, times),axis=0)).float() # (2, num_fr)
        manifold = manifold.permute(1,0) # (num_fr, 2) 
    else:
        raise NotImplementedError("No such input_type.")
    return manifold

def get_input_manifold_2(input_type, motion_index):
    if input_type.startswith('2Dline'):
        #motion_indices size([1,1600])
        #times size([1,1600])
        manifold = torch.tensor(motion_index).float() # (2, num_fr)
    else:
        raise NotImplementedError("No such input_type.")
    return manifold

def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.
    
    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.

def np_to_pil(img_np): 
    '''Converts image in np.array format to PIL image.
    
    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np*255,0,255).astype(np.uint8)
    
    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]


def ifft_maskc_fft(input, maskc, coilmap):
    #coilmap = coilmap.unsqueeze(0) # (batch, real-cmplx, size_x, size_y, ncoil)
    (_, _, size_x, size_y) = input.shape
    (_, _, size_mask_x, size_mask_y) = input.shape
    (_, _, size_x, size_y, ncoil) = coilmap.shape
    tmp = []
    for i in range(coilmap.size(dim=4)):
        coil_img = th_mul_cplx(input, coilmap[:,:,:,:,i])
        kspace = fftc_th(zeropad_batch(coil_img, (size_mask_x, size_mask_y)))
        kspace_c = kspace*maskc
        coil_img_2 = th_mul_cplx(cropcenter_batch(ifftc_th(kspace_c),(size_x, size_y)),th_conj(coilmap[:,:,:,:,i]))
        tmp.append(coil_img_2)
    output = torch.stack(tmp,dim=4).sum(dim=4)
    return output


def th_mul_cplx(e,f):
    #a,b is [(batch, real-cmplx, size_x, size_y]
    a = e[:,0:1,:,:]
    b = e[:,1:2,:,:]
    c = f[:,0:1,:,:]
    d = f[:,1:2,:,:]
    return torch.cat((a*c - b*d, a*d + b*c),1)



def fftc_th(image):
    image = image.permute(0, 2, 3, 1).contiguous()
    kspace = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(torch.view_as_complex(image), dim=(-1, -2)),norm = "ortho" ), dim=(-1, -2))
    kspace = torch.view_as_real(kspace).permute(0, 3, 1, 2).contiguous()
    return kspace


def ifftc_th(kspace):
    kspace = kspace.permute(0, 2, 3, 1).contiguous()
    image = torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(torch.view_as_complex(kspace), dim=(-1, -2)),norm = "ortho" ), dim=(-1, -2))
    image = torch.view_as_real(image).permute(0, 3, 1, 2).contiguous()
    return image




def gradL1Reg(x):
    Dx = torch.cat((x[1:,:], x[-1:,:]),0) - x
    reg = torch.abs(Dx).sum() 
    return reg



def gradL2Reg(x):
    Dt = torch.cat((x[1:,:,:,:], x[-1:,:,:,:]),0) - x
    reg = torch.sum(Dt^2) 
    return reg

def gradL2RegDfield(img):
    #img: torch.Size([batch_size, nslice, 256, 256, 3])

    Dx = torch.cat((img[:,1:,:,:,:], img[:,-1:,:,:,:]), dim=1) - img
    Dy = torch.cat((img[:,:,1:,:,:], img[:,:,-1:,:,:]), dim=2) - img
    Dz  = torch.cat((img[:,:,:,1:,:], img[:,:,:,-1:,:]), dim=3) - img
    reg = torch.square(Dx).sum() + torch.square(Dy).sum() + torch.square(Dz).sum()
    return reg