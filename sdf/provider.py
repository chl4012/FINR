import numpy as np

import torch
from torch.utils.data import Dataset
import random
from utils.common_utils import *
from utils.nufft_utils import *
import torch.nn.functional as F
import scipy.io as sio

class HashGrid3D_MRF_fat_water_Dataset(Dataset):
    def __init__(self, kspaceMRF, T, index_all, par, motion_values, Ur_full_list, slice_full_dict, ktraj_view_list, size=100, flat_flag = False):
        super().__init__()
        self.dataMRF = kspaceMRF
        self.T = T
        self.index_all = index_all
        self.motion_values = motion_values
        self.size = size
        self.Ur_full_list = Ur_full_list
        self.slice_full_dict = slice_full_dict
        self.ktraj_view_list = ktraj_view_list
        self.par = par
        Nslice = par['Nslice']
        self.Nslice = Nslice
        self.Necho = par['Necho']
        self.flat_flag = flat_flag

    def __len__(self):
        return self.size

    def __getitem__(self, idx_bin):
        par = self.par
        T1_stacks = par['T1_stacks']
        MGRE_stacks_ = par['MGRE_stacks_']
        T2_stacks = par['T2_stacks']
        nslice_se = par['nslice_se']
        nslice_me = par['nslice_me']
        Nrep = par["Nrep"]

        #map_in = torch.tensor(np.mean(self.motion_values[:,self.size-1])).to(dtype =torch.float32) - torch.tensor(np.mean(self.motion_values[:,idx_bin])).to(dtype =torch.float32) #(1)
        map_in = torch.tensor(np.mean(self.motion_values[:,idx_bin])).to(dtype =torch.float32) #(1)
        motion01_list = get_motion01_list( T1_stacks, MGRE_stacks_, T2_stacks,
                      Nrep, nslice_se, nslice_me,self.index_all, idx_bin) #size(32000,)
        

        #echo_indices.sort()
        bin_indices = get_bin_MRF_indices(T1_stacks, MGRE_stacks_, T2_stacks, 
                  Nrep, nslice_se, nslice_me, motion01_list) #size(32000/binnum,)
        Ur_bin_list = torch.tensor(self.Ur_full_list[bin_indices,:])
        dataMRF_bin = torch.tensor(self.dataMRF[:,:,bin_indices,:]).squeeze().to(dtype = torch.complex64) 
        bin_ktraj_views= self.ktraj_view_list[bin_indices]
        ktraj = torch.tensor(np.stack((np.real(self.T[bin_ktraj_views,:]), 
                              np.imag(self.T[bin_ktraj_views,:])), axis=2)).to(dtype = torch.float32)
        slice_bin_list = self.slice_full_dict[bin_indices].flatten()
        w = 256
        h = 256
        d = self.Nslice
        locations_x = torch.linspace(0, w-1, w).view(w, 1,1).expand(w, h,d).contiguous().view(-1,1)
        locations_y = torch.linspace(0, h-1, h).view(1, h,1).expand(w, h,d).contiguous().view(-1,1)
        locations_z = torch.linspace(0, d-1, d).view(1, 1,d).expand(w, h,d).contiguous().view(-1,1)
        locations_x = (2.0*locations_x - (w-1))/(w-1)
        locations_y = (2.0*locations_y - (h-1))/(h-1)
        locations_z = (2.0*locations_z - (d-1))/(d-1)
        # stack locations
        points_x = torch.stack([locations_x, locations_y, locations_z], dim=1)
        locations_t = map_in*torch.ones(w, h,d).contiguous().view(-1,1)
        points_xt = torch.stack([locations_x, locations_y, locations_z, locations_t], dim=1)

        results = {
            'points_x': points_x,
            'points_xt': points_xt,
            'ktraj': ktraj,
            'Ur_bin_list':Ur_bin_list,
            'slice_bin_list':slice_bin_list,
            'dataMRF_bin': torch.view_as_real(dataMRF_bin),
            'map_in': map_in,
            'idx_bin': idx_bin,
        }
        
        return results
    
