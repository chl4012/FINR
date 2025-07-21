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
import os 
import mat73
from math import pi
from sigpy.mri.app import EspiritCalib



def get_index_all(motion_values, Nrep, Ncontrast, window_size):
    #get the indices related to all different motion states 
    motion_indices = motion_values.reshape(Nrep,Ncontrast)
    sorted_indices = motion_indices.argsort(0)
    new_position = sorted_indices.argsort(0)
    motion_indices = new_position
    numbins = int(Nrep-window_size+1)
    index_all = [] 
    for b in range(numbins):
        index_bin = []
        for i in range(Ncontrast):
            index = np.arange(Nrep) * Ncontrast + i
            bin_b_idx = (motion_indices[:,i] >= b) & (motion_indices[:,i] < b  + window_size)
            index_values = np.nonzero(bin_b_idx)
            index = index[index_values].reshape(1,-1)
            index_bin.append(index)
        index_bin = np.concatenate(index_bin, axis = 0)
        index_all.append(index_bin)
    index_all = np.stack(index_all, axis = 2)
    index_all_2 = np.reshape(index_all,(-1,numbins))
    return index_all, index_all_2



def get_index_all_non_overlap(motion_values, Nrep, Ncontrast, window_size):
    #get the indices related to all different motion states 
    motion_indices = motion_values.reshape(Nrep,Ncontrast)
    sorted_indices = motion_indices.argsort(0)
    new_position = sorted_indices.argsort(0)
    motion_indices = new_position
    numbins = int(Nrep/window_size)
    index_all = [] 
    for b in range(numbins):
        index_bin = []
        for i in range(Ncontrast):
            index = np.arange(Nrep) * Ncontrast + i
            bin_b_idx = (motion_indices[:,i] >= b*window_size) & (motion_indices[:,i] < (b+1)*window_size)
            index_values = np.nonzero(bin_b_idx)
            index = index[index_values].reshape(1,-1)
            index_bin.append(index)
        index_bin = np.concatenate(index_bin, axis = 0)
        index_all.append(index_bin)
    index_all = np.stack(index_all, axis = 2)
    index_all_2 = np.reshape(index_all,(-1,numbins))
    return index_all, index_all_2


def get_motion01_list(T1_stacks, MGRE_stacks_, T2_stacks,
                      numrep, nslice_se, nslice_me, index_all, idx_bin):
    index_bin = index_all[:,:,idx_bin]
    motion01_full = []
    startline_T1 = 0
    endline_T1 = T1_stacks
    startline_MGRE1 = endline_T1 
    endline_MGRE2 = endline_T1 + MGRE_stacks_ 
    startline_T2 = endline_MGRE2 
    endline_T2 = endline_MGRE2 + T2_stacks
    for i in range(numrep * endline_T2):  # Python ranges start at 0, hence the +1
        k = i % endline_T2  # Adjust for Python's 0-based indexing
        if startline_T1 <= k < endline_T1:
            if i in list(index_bin[k,:]):
                motion01_full.append([1]*nslice_se)
            else:
                motion01_full.append([0]*nslice_se)
        if startline_MGRE1 <= k < endline_MGRE2:
            if i in list(index_bin[k,:]):
                motion01_full.append([1]*nslice_me)
            else:
                motion01_full.append([0]*nslice_me)
        if startline_T2 <= k < endline_T2:
            if i in list(index_bin[k,:]):
                motion01_full.append([1]*nslice_se)
            else:
                motion01_full.append([0]*nslice_se)
    return np.concatenate(motion01_full,0)


def get_bin_MRF_indices(T1_stacks, MGRE_stacks_, T2_stacks, 
                      numrep, nslice_se, nslice_me, motion01_list):
    motion01_list = np.nonzero(motion01_list)[0]
    motion01_list.tolist()  # Convert the result back to a Python list if needed
    bin_MRF_indices = []
    # Set the starting point for MGRE2
    endline_T2 = MGRE_stacks_*nslice_me + (T1_stacks + T2_stacks) * nslice_se
    # Loop through the number of repetitions and contrasts
    for i in range(numrep * endline_T2):  # Python ranges start at 0, hence the +1
        if i in motion01_list:
            bin_MRF_indices.append(i)
    return np.array(bin_MRF_indices)




#take one stack as a different contrast 
def get_Ur_list(T1_stacks, MGRE_stacks_, T2_stacks, 
                      numrep, nslice_se, nslice_me, Ur):
    #Ur: size(Ncontrast, Nsingulars)
    Ur_list = []
    startline_T1 = 0
    endline_T1 = T1_stacks
    startline_MGRE1 = endline_T1 
    endline_MGRE2 = endline_T1 + MGRE_stacks_ 
    startline_T2 = endline_MGRE2 
    endline_T2 = endline_MGRE2 + T2_stacks
    for i in range(numrep * endline_T2):  # Python ranges start at 0, hence the +1
        k = i % endline_T2  # Adjust for Python's 0-based indexing
        if startline_T1 <= k < endline_T1:
            for tt in range(nslice_se):
                Ur_list.append(Ur[[k],:])
        if startline_MGRE1 <= k < endline_MGRE2:
            for tt in range(nslice_me):
                Ur_list.append(Ur[[k],:])
        if startline_T2 <= k < endline_T2:
            for tt in range(nslice_se):
                Ur_list.append(Ur[[k],:])
    return np.concatenate(Ur_list,0)


#take each TR as a differnt contrast
def get_Ur_list_v2(T1_stacks, MGRE_stacks_, T2_stacks, 
                      numrep, nslice_se, nslice_me, Ur):
    #Ur: size(Ncontrast, Nsingulars)
    Ur_list = []
    startline_T1 = 0
    endline_T1 = T1_stacks*nslice_se
    startline_MGRE1 = endline_T1 
    endline_MGRE2 = endline_T1 + MGRE_stacks_*nslice_me
    startline_T2 = endline_MGRE2 
    endline_T2 = endline_MGRE2 + T2_stacks*nslice_se
    for i in range(numrep * endline_T2):  # Python ranges start at 0, hence the +1
        k = i % endline_T2  # Adjust for Python's 0-based indexing
        if startline_T1 <= k < endline_T1:
            Ur_list.append(Ur[[k],:])
        if startline_MGRE1 <= k < endline_MGRE2:
            Ur_list.append(Ur[[k],:])
        if startline_T2 <= k < endline_T2:
            Ur_list.append(Ur[[k],:])
    return np.concatenate(Ur_list,0)



def get_ktraj_view_list(T1_stacks, MGRE_stacks, T2_stacks, numrep, nslice_se):
    ktraj_view_list = []
    startline_T1 = 0
    endline_T1 = T1_stacks
    startline_MGRE1 = endline_T1 
    endline_MGRE2 = endline_T1 + MGRE_stacks
    startline_T2 = endline_MGRE2 
    endline_T2 = endline_MGRE2 + T2_stacks
    for i in range(numrep * endline_T2):  # Python ranges start at 0, hence the +1
        ktraj_view_list.append([i]*nslice_se)
    return np.concatenate(ktraj_view_list,0)

