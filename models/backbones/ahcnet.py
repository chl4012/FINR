import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from torch.autograd import Variable
from torch.nn import Parameter

 # type: GN, BN, IN, LN
 # Specifically, GN with GN GN_2 GN_4 ...
def getNormModuleND(dimension = 3, norm_type = 'GN'):

    if norm_type is None:
        return None, None

    norm_ND = None
    params = {}

    if 'GN' in norm_type:
        norm_ND = nn.GroupNorm
        if norm_type == 'GN':
            params['num_groups'] = 2
        else:
            eles = norm_type.split('_')
            params['num_groups'] = int(eles[1])  
    elif 'IN' in norm_type:
        norm_ND = nn.InstanceNorm3d
        if dimension == 1:
            norm_ND = nn.InstanceNorm1d
        elif dimension == 2:
            norm_ND = nn.InstanceNorm2d
        elif dimension == 3:
            norm_ND = nn.InstanceNorm3d
        params['affine'] = True
    else:
        if dimension == 1:
            norm_ND = nn.BatchNorm1d
        elif dimension == 2:
            norm_ND = nn.BatchNorm2d
        elif dimension == 3:
            norm_ND = nn.BatchNorm3d
    
    return norm_ND, params
     
class convNormAct(nn.Module):

    def __init__(self, in_cs, out_cs, norm_type, kernel_size, stride, padding, p_mode='same', is_act=True):
    
        super(convNormAct, self).__init__()

        norm_3d, norm_params = getNormModuleND(dimension = 3, norm_type = norm_type)
        if p_mode == 'same':
            self.conv =  nn.Conv3d(in_cs, out_cs, kernel_size, stride, padding, bias = False)
        if p_mode == 'sym':
            self.conv =  nn.Conv3d(in_cs, out_cs, kernel_size, stride, 0, bias = False)

        if norm_type is not None:
            if 'GN' in norm_type:
                norm_params['num_channels'] = out_cs
            else:
                norm_params['num_features'] = out_cs
            self.norm = norm_3d(**norm_params)
        else:
            self.norm = nn.Sequential()
        
        self.relu = nn.ReLU(inplace=True)
        self.p_mode = p_mode
        self.padding = padding
        self.is_act = is_act

    def forward(self, x):
        
        if self.p_mode == 'sym' and self.padding == 1:
            x = F.pad(x, (1,1,1,1,1,1), mode='replicate')
        x = self.conv(x)
        x = self.norm(x)
        if self.is_act:
            x = self.relu(x)
        
        return x

class normActConv(nn.Module):
    
    def __init__(self, in_cs, out_cs, norm_type, kernel_size, stride, padding, p_mode='same'):

        super(normActConv, self).__init__()

        norm_3d, norm_params = getNormModuleND(dimension = 3, norm_type = norm_type)
        if p_mode == 'same':
            self.conv =  nn.Conv3d(in_cs, out_cs, kernel_size, stride, padding, bias = False)
        if p_mode == 'sym':
            self.conv =  nn.Conv3d(in_cs, out_cs, kernel_size, stride, 0, bias = False)

        if norm_type is not None:
            if 'GN' in norm_type:
                norm_params['num_channels'] = in_cs
            else:
                norm_params['num_features'] = in_cs
            self.norm = norm_3d(**norm_params)
        else:
            self.norm = nn.Sequential()
        
        self.relu = nn.ReLU(inplace=True)
        self.p_mode = p_mode
        self.padding = padding

    def forward(self, x):
        
        if self.p_mode == 'sym' and self.padding == 1:
            x = F.pad(x, (1,1,1,1,1,1), mode='replicate')

        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        
        return x

class unetResBlockV1(nn.Module):

    def __init__(self, in_cs, out_cs, norm_type, p_mode='same', downsample=False):

        super(unetResBlockV1, self).__init__()

        stride = 2 if downsample else 1

        if in_cs != out_cs or stride != 1:
            norm_3d, norm_params = getNormModuleND(dimension = 3, norm_type = norm_type)
            if norm_type is not None:
                if 'GN' in norm_type:
                    norm_params['num_channels'] = out_cs
                else:
                    norm_params['num_features'] = out_cs
                self.shortcut = nn.Sequential(
                    nn.Conv3d(in_cs, out_cs, 1, stride, 0, bias = False),
                    norm_3d(**norm_params),
                )
            else:
                self.shortcut = nn.Conv3d(in_cs, out_cs, 1, stride, 0, bias = False)
        else:
            self.shortcut = nn.Sequential() 

        self.layer1 = convNormAct(in_cs, out_cs, norm_type, 3, stride, 1, is_act = True)
        self.layer2 = convNormAct(out_cs, out_cs, norm_type, 3, 1, 1, is_act = False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        identity = x

        x = self.layer2(self.layer1(x))
        x = x + self.shortcut(identity)
        x = self.relu(x)
        
        return x

class unetVggBlock(nn.Module):

    def __init__(self, in_cs, out_cs, norm_type, p_mode='same', num_convs = 2): 

        super(unetVggBlock, self).__init__()

        convs = []
        convs.append(convNormAct(in_cs, out_cs, norm_type, 3, 1, 1, p_mode))
        for _ in range(num_convs-1):
            convs.append(convNormAct(out_cs, out_cs, norm_type, 3, 1, 1, p_mode))

        self.layer = nn.Sequential(*convs)

    def forward(self, x):

        return self.layer(x)



class unetVggBlockShortCut(nn.Module):

    def __init__(self, in_cs, out_cs, norm_type, p_mode='same'): 

        super(unetVggBlockShortCut, self).__init__()

        self.conv1=convNormAct(in_cs, out_cs, norm_type, 3, 1, 1, p_mode)
        self.conv2=convNormAct(out_cs, out_cs, norm_type, 3, 1, 1, p_mode)

        
    def forward(self, x):

        x1=self.conv1(x)
        x2=self.conv2(x1)

        return x1+x2


class convMultiN(nn.Module):

    def __init__(self, in_cs, norm_type, kernel_size, stride, dilation):

        super(convMultiN, self).__init__()

        self.conv1 = nn.Conv3d(in_cs   , in_cs//2, kernel_size, stride, dilation, dilation, bias = False)
        self.conv2 = nn.Conv3d(in_cs//2, in_cs//4, kernel_size, stride, dilation, dilation, bias = False)
        self.conv3 = nn.Conv3d(in_cs//4, in_cs//4, kernel_size, stride, dilation, dilation, bias = False)
        
        norm_3d, norm_params = getNormModuleND(dimension = 3, norm_type = norm_type)
        
        self.final_conv = nn.Sequential(
            nn.Conv3d(in_cs*2, in_cs*2, 1, 1, 0, bias = False),
            norm_3d(*(norm_params + [in_cs*2])),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x1):
        
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)

        x = torch.cat([x1,x2,x3,x4],dim=1)

        x = self.final_conv(x)

        return x

class unetVggBlockMultiN(nn.Module):

    def __init__(self, in_cs, out_cs, norm_type, padding=1, dilation=1, p_mode='same'):

        super(unetVggBlockMultiN, self).__init__()

        self.conv1 = convNormAct(in_cs, out_cs // 2, norm_type, 3, 1, padding, p_mode=p_mode)
        self.conv2 = convMultiN(out_cs // 2, norm_type, 3, 1, dilation=dilation)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)

        return x

def _bn_function_factory(norm, relu, conv):
    
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function




class unetConvBlock(nn.Module):
    
    '''
    Three types of convolutions are supported.
    vgg: with control of number of convolution (num_convs, default 2) in one vgg conv layer
    resV1: resnet with post activation
    resV2: resnet with pre activation
    '''

    def __init__(self, in_cs, out_cs, norm_type, num_convs = 2, conv_type = 'vgg', p_mode = 'same'): 

        super(unetConvBlock, self).__init__()

        if conv_type == 'vgg':
            self.layer = unetVggBlock(in_cs, out_cs, norm_type, p_mode, num_convs)
        elif conv_type == 'resV1':
            self.layer = unetResBlockV1(in_cs, out_cs, norm_type, p_mode)
        elif 'dense' in conv_type:
            num_blocks = int(conv_type.split('_')[1])
            self.layer = fullDensenBlock(in_cs, out_cs, norm_type, num_blocks=num_blocks)
        elif conv_type == 'pre_dense':
            self.layer = preDensenBlock(in_cs, out_cs, norm_type)
        elif conv_type == 'vgg_multi_d':
            self.layer = unetVggBlockMultiN(in_cs, out_cs, norm_type, padding=1, dilation=2, p_mode=p_mode)
        elif conv_type == 'vgg_multi_n':
            self.layer = unetVggBlockMultiN(in_cs, out_cs, norm_type, padding=1, dilation=1, p_mode=p_mode)
            
    def forward(self, x):

        return self.layer(x)

class unetDownSample(nn.Module):

    def __init__(self, channels, down_type = 'conv', p_mode = 'same', norm_type = None):

        super(unetDownSample, self).__init__()

        if down_type == 'conv':
            if p_mode == 'same':
                self.down = nn.Conv3d(channels, channels, 3, 2, 1, bias = False)
            elif p_mode == 'sym':
                self.down = nn.Conv3d(channels, channels, 3, 2, 0, bias = False)
        if down_type == 'maxpool':
            self.down = nn.MaxPool3d(kernel_size = 2, padding = 0)
        
        self.p_mode = p_mode
        self.down_type = down_type

    def forward(self, x):
        
        if self.p_mode == 'sym' and self.down_type != 'maxpool':
            x = F.pad(x, (1,1,1,1,1,1), mode='replicate')

        return self.down(x)

class unetUpConv(nn.Module):

    def __init__(self, in_cs, out_cs, is_deconv, upsample_type, norm_type, conv_type):

        super(unetUpConv, self).__init__()

        layers = []
        if norm_type is not None:
            if 'pre' in conv_type:
                layers.append(convNormAct(in_cs,out_cs,norm_type,1,1,0))
            else:
                layers.append(normActConv(in_cs,out_cs,norm_type,1,1,0))
        else:
            layers.append(nn.Conv3d(in_cs, out_cs, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.ReLU(inplace=True))
       
        if is_deconv:
            layers.append(nn.ConvTranspose3d(out_cs, out_cs, kernel_size=2, stride=2, padding=0, bias=False)) 
        else:
            if upsample_type == 'nearest':
                layers.append(nn.Upsample(scale_factor=2, mode = upsample_type))
            else:
                layers.append(nn.Upsample(scale_factor=2, mode = upsample_type, align_corners = True))

        self.up = nn.Sequential(*layers)
    
    def forward(self, x):
        
        return self.up(x)

class unetPadCat(nn.Module):

    def __init__(self):

        super(unetPadCat, self).__init__()

    def forward(self, leftIn, rightIn):
        
        rShape = rightIn.size()
        lShape = leftIn.size()
        padding = (lShape[4]-rShape[4], 0, lShape[3]-rShape[3], 0, lShape[2]-rShape[2], 0)

        pad = torch.nn.ConstantPad3d(padding, 0)
        rightIn = pad(rightIn)

        return torch.cat([leftIn, rightIn], 1)

class unetPad(nn.Module):

    def __init__(self):

        super(unetPad, self).__init__()

    def forward(self, leftIn, rightIn):
        
        rShape = rightIn.size()
        lShape = leftIn.size()
        padding = (lShape[4]-rShape[4], 0, lShape[3]-rShape[3], 0, lShape[2]-rShape[2], 0)

        pad = torch.nn.ConstantPad3d(padding, 0)
        rightIn = pad(rightIn)

        return rightIn

class unetUpPadCatConv(nn.Module):
    
    def __init__(self, left_cs, right_cs, is_deconv, norm_type, p_mode='same', conv_type='vgg', upsample_type='nearest'):

        super(unetUpPadCatConv, self).__init__()

        self.up = unetUpConv(right_cs, right_cs // 2, is_deconv, upsample_type, norm_type, conv_type)
        self.pad = unetPad()
        self.conv = unetConvBlock(right_cs, left_cs, norm_type, conv_type = conv_type, p_mode=p_mode)
    
    def forward(self, left_x, right_x):

        right_x = self.up(right_x) 
        right_x = self.pad(left_x, right_x)
        x=torch.cat([left_x, right_x], 1)
        x = self.conv(x)
        del left_x

        return x

class unetUpPad(nn.Module):
    
    def __init__(self, right_cs, is_deconv, norm_type, p_mode='same', conv_type='vgg', upsample_type='nearest'):

        super(unetUpPad, self).__init__()

        self.up = unetUpConv(right_cs, right_cs // 2, is_deconv, upsample_type, norm_type, conv_type)
        self.pad = unetPad()
    
    def forward(self, left_x, right_x):

        right_x = self.up(right_x) 
        right_x = self.pad(left_x, right_x)
        del left_x

        return right_x

class unetUpConvBlock(nn.Module):
    
    def __init__(self, left_cs, right_cs, is_deconv, norm_type, p_mode='same', conv_type='vgg', upsample_type='nearest'):

        super(unetUpConvBlock, self).__init__()

        self.up = unetUpConv(right_cs, right_cs , is_deconv, upsample_type, norm_type, conv_type)
        self.pad = unetPad()
        self.conv = unetConvBlock(right_cs, left_cs, norm_type, conv_type = conv_type, p_mode=p_mode)
    
    def forward(self, left_x, right_x):

        right_x = self.up(right_x) 
        x = self.pad(left_x, right_x)
        del left_x, right_x

        return self.conv(x)

class AttentionGate(nn.Module):
    
    def __init__(self,channel):

        super(AttentionGate, self).__init__()
        
        self.block=nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(channel,channel,1,1,0),
            nn.Tanh()
        )

    def forward(self, left_x, right_x):

        x=left_x+right_x
        x=self.block(x)
        del right_x

        return left_x + x
