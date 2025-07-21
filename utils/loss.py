import torch

def ncc_loss(I1, I2):
    #normalized cross correlation
    # Mean of the images
    I1_mean = torch.mean(I1)
    I2_mean = torch.mean(I2)
    # Standard deviations of the images
    I1_std = torch.std(I1)
    I2_std = torch.std(I2)
    # Calculate the numerator (covariance)
    numerator = torch.sum((I1 - I1_mean) * (I2 - I2_mean))
    # Calculate the denominator (product of standard deviations)
    denominator = I1_std * I2_std
    # Number of elements in the images
    N = torch.numel(I1)
    # Calculate the NCC
    ncc = numerator / (denominator * N)
    
    return ncc


import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn
import math

class CustomLoss(nn.Module):
    def __init__(self, w_mean_r, w_std_r, f_mean_r, f_std_r, w_mean_i, w_std_i, f_mean_i, f_std_i, frq_mean, frq_std, r2_mean, r2_std):
        super(CustomLoss, self).__init__()
        self.w_mean_r = w_mean_r
        self.w_std_r = w_std_r
        self.f_mean_r = f_mean_r
        self.f_std_r = f_std_r
        self.w_mean_i = w_mean_i
        self.w_std_i = w_std_i
        self.f_mean_i = f_mean_i
        self.f_std_i = f_std_i
        self.frq_mean = frq_mean
        self.frq_std = frq_std
        self.r2_mean = r2_mean
        self.r2_std = r2_std
        self.pi = math.pi

    def forward(self, input_train_tm_tens, input_train_tp_tens, dfat_train_t_tens, te_train_t_tens, output_pred):
        wat_r = output_pred[..., 0]
        watt_r = wat_r.unsqueeze(-1).repeat(1, 1, 1, 6)
        watt_r = watt_r * self.w_std_r + self.w_mean_r

        fat_r = output_pred[..., 1]
        fatt_r = fat_r.unsqueeze(-1).repeat(1, 1, 1, 6)
        fatt_r = fatt_r * self.f_std_r + self.f_mean_r

        wat_i = output_pred[..., 3]
        watt_i = wat_i.unsqueeze(-1).repeat(1, 1, 1, 6)
        watt_i = watt_i * self.w_std_i + self.w_mean_i

        fat_i = output_pred[..., 4]
        fatt_i = fat_i.unsqueeze(-1).repeat(1, 1, 1, 6)
        fatt_i = fatt_i * self.f_std_i + self.f_mean_i

        frq = output_pred[..., 5]
        frqt = frq.unsqueeze(-1).repeat(1, 1, 1, 6)
        frqt = frqt * self.frq_std + self.frq_mean

        r2 = output_pred[..., 2]
        r2t = r2.unsqueeze(-1).repeat(1, 1, 1, 6)
        r2t = r2t * self.r2_std + self.r2_mean

        # PyTorch does not have a direct complex tensor before version 1.6.0, but you can use torch.complex in newer versions
        watt_c = torch.complex(watt_r, torch.zeros_like(watt_r))
        fatt_c = torch.complex(fatt_r, torch.zeros_like(fatt_r))

        watt_ci = torch.complex(torch.zeros_like(watt_i), watt_i)
        fatt_ci = torch.complex(torch.zeros_like(fatt_i), fatt_i)

        r2t_c = torch.complex(r2t, torch.zeros_like(r2t))
        frqt_c = torch.complex(frqt, torch.zeros_like(frqt))
        dfat_train_t_c = torch.complex(dfat_train_t_tens, torch.zeros_like(dfat_train_t_tens))
        te_train_t_c = torch.complex(te_train_t_tens, torch.zeros_like(te_train_t_tens))

        signal = (watt_c * torch.exp(watt_ci) + fatt_c * torch.exp(fatt_ci - 1j * 2 * self.pi * dfat_train_t_c * te_train_t_c)) * torch.exp(-1 * r2t_c * te_train_t_c) * torch.exp(-1j * 2 * self.pi * frqt_c * te_train_t_c)

        input_train_t_mag = input_train_tm_tens[..., 0:6]
        input_train_t_phs = input_train_tp_tens[..., 0:6]

        gt_input_train2 = torch.complex(input_train_t_mag, torch.zeros_like(input_train_t_mag)) * torch.exp(torch.complex(torch.zeros_like(input_train_t_phs), input_train_t_phs))

        loss = torch.sqrt(torch.sum(torch.abs(gt_input_train2 - signal) ** 2))
        return loss
