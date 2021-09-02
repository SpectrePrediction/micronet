from os import replace
from pickle import DICT
import re
from pyparsing import CaselessLiteral, Forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import BACKBONES
from torch.autograd import Variable
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import mmcv

import math

_MODEL_URLS = {
    "": "",
}

_MODEL_CFG = {
    'm0': dict(
        net_config='msnx_dy6_exp4_4M_221',
        block='DYMicroBlock',
        stem_mode='spatialsepsf',
        stem_ch=4,
        stem_dilation=1,
        stem_groups=[2, 2],
        out_ch=640,
        depthsep=True,
        shuffle=True,
        pointwise='group',
        dropout=0.05,
        # defind
        activation=dict(
            module='DYShiftMax',
            act_max=2.0,
            linearse_bias=False,
            init_a_block3=[1.0, 0.0],
            init_a=[1.0, 1.0],
            init_b=[0.0, 0.0],
            reduction=8,
            last_se_oup=False,
            fc=False,
            act='relu',
        )
    ),
    'm1': dict(
        net_config='msnx_dy6_exp6_6M_221',
        block='DYMicroBlock',
        stem_mode='spatialsepsf',
        stem_ch=6,
        stem_dilation=1,
        stem_groups=[3, 2],
        out_ch=960,
        depthsep=True,
        shuffle=True,
        pointwise='group',
        dropout=0.05,
        # defind
        activation=dict(
            module='DYShiftMax',
            act_max=2.0,
            linearse_bias=False,
            init_a_block3=[1.0, 0.0],
            init_a=[1.0, 1.0],
            init_b=[0.0, 0.0],
            reduction=8,
            last_se_oup=False,
            fc=False,
            act='relu',
        )
    ),
    'm2': dict(
        net_config='msnx_dy9_exp6_12M_221',
        block='DYMicroBlock',
        stem_mode='spatialsepsf',
        stem_ch=8,
        stem_dilation=1,
        stem_groups=[4, 2],
        out_ch=1024,
        depthsep=True,
        shuffle=True,
        pointwise='group',
        dropout=0.1,
        # defind
        activation=dict(
            module='DYShiftMax',
            act_max=2.0,
            linearse_bias=False,
            init_a_block3=[1.0, 0.0],
            init_a=[1.0, 1.0],
            init_b=[0.0, 0.0],
            reduction=8,
            last_se_oup=False,
            fc=False,
            act='relu',
        )
    ),
    'm3': dict(
        net_config='msnx_dy12_exp6_20M_020',
        block='DYMicroBlock',
        stem_mode='spatialsepsf',
        stem_ch=12,
        stem_dilation=1,
        stem_groups=[4, 3],
        out_ch=1024,
        depthsep=True,
        shuffle=True,
        pointwise='group',
        dropout=0.1,
        # defind
        activation=dict(
            module='DYShiftMax',
            act_max=2.0,
            linearse_bias=False,
            init_a_block3=[1.0, 0.0],
            init_a=[1.0, 0.5],
            init_b=[0.0, 0.5],
            reduction=8,
            last_se_oup=False,
            fc=False,
            act='relu',
        )
    ),

}

msnx_dy6_exp4_4M_221_cfgs = [
        #s, n,  c, ks, c1, c2, g1, g2, c3, g3, g4,y1,y2,y3,r
        [2, 1,   8, 3, 2, 2,  0,  4,   8,  2,  2, 2, 0, 1, 1],  #6->12(0, 0)->24  ->8(4,2)->8
        [2, 1,  12, 3, 2, 2,  0,  8,  12,  4,  4, 2, 2, 1, 1], #8->16(0, 0)->32  ->16(4,4)->16
        [2, 1,  16, 5, 2, 2,  0, 12,  16,  4,  4, 2, 2, 1, 1], #16->32(0, 0)->64  ->16(8,2)->16
        [1, 1,  32, 5, 1, 4,  4,  4,  32,  4,  4, 2, 2, 1, 1], #16->16(2,8)->96 ->32(8,4)->32
        [2, 1,  64, 5, 1, 4,  8,  8,  64,  8,  8, 2, 2, 1, 1], #32->32(2,16)->192 ->64(12,4)->64
        [1, 1,  96, 3, 1, 4,  8,  8,  96,  8,  8, 2, 2, 1, 2], #64->64(3,16)->384 ->96(16,6)->96
        [1, 1, 384, 3, 1, 4, 12, 12,   0,  0,  0, 2, 2, 1, 2], #96->96(4,24)->576
]

msnx_dy6_exp6_6M_221_cfgs = [
        #s, n,  c, ks, c1, c2, g1, g2, c3, g3, g4
        [2, 1,   8, 3, 2, 2,  0,  6,   8,  2,  2, 2, 0, 1, 1],  #6->12(0, 0)->24  ->8(4,2)->8
        [2, 1,  16, 3, 2, 2,  0,  8,  16,  4,  4, 2, 2, 1, 1], #8->16(0, 0)->32  ->16(4,4)->16
        [2, 1,  16, 5, 2, 2,  0, 16,  16,  4,  4, 2, 2, 1, 1], #16->32(0, 0)->64  ->16(8,2)->16
        [1, 1,  32, 5, 1, 6,  4,  4,  32,  4,  4, 2, 2, 1, 1], #16->16(2,8)->96 ->32(8,4)->32
        [2, 1,  64, 5, 1, 6,  8,  8,  64,  8,  8, 2, 2, 1, 1], #32->32(2,16)->192 ->64(12,4)->64
        [1, 1,  96, 3, 1, 6,  8,  8,  96,  8,  8, 2, 2, 1, 2], #64->64(3,16)->384 ->96(16,6)->96
        [1, 1, 576, 3, 1, 6, 12, 12,   0,  0,  0, 2, 2, 1, 2], #96->96(4,24)->576
]

msnx_dy9_exp6_12M_221_cfgs = [
        #s, n,  c, ks, c1, c2, g1, g2, c3, g3, g4
        [2, 1,  12, 3, 2, 2,  0,  8,  12,  4,  4, 2, 0, 1, 1], #8->16(0, 0)->32  ->12(4,3)->12
        [2, 1,  16, 3, 2, 2,  0, 12,  16,  4,  4, 2, 2, 1, 1], #12->24(0,0)->48  ->16(8, 2)->16
        [1, 1,  24, 3, 2, 2,  0, 16,  24,  4,  4, 2, 2, 1, 1], #16->16(0, 0)->64  ->24(8,3)->24
        [2, 1,  32, 5, 1, 6,  6,  6,  32,  4,  4, 2, 2, 1, 1], #24->24(2, 12)->144  ->32(16,2)->32
        [1, 1,  32, 5, 1, 6,  8,  8,  32,  4,  4, 2, 2, 1, 2], #32->32(2,16)->192 ->32(16,2)->32
        [1, 1,  64, 5, 1, 6,  8,  8,  64,  8,  8, 2, 2, 1, 2], #32->32(2,16)->192 ->64(12,4)->64
        [2, 1,  96, 5, 1, 6,  8,  8,  96,  8,  8, 2, 2, 1, 2], #64->64(4,12)->384 ->96(16,5)->96
        [1, 1, 128, 3, 1, 6, 12, 12, 128,  8,  8, 2, 2, 1, 2], #96->96(5,16)->576->128(16,8)->128
        [1, 1, 768, 3, 1, 6, 16, 16,   0,  0,  0, 2, 2, 1, 2], #128->128(4,32)->768
]

msnx_dy12_exp6_20M_020_cfgs = [
    #s, n,  c, ks, c1, c2, g1, g2, c3, g3, g4
    [2, 1,  16, 3, 2, 2,  0, 12,  16,  4,  4, 0, 2, 0, 1], #12->24(0, 0)->48  ->16(8,2)->16
    [2, 1,  24, 3, 2, 2,  0, 16,  24,  4,  4, 0, 2, 0, 1], #16->32(0, 0)->64  ->24(8,3)->24
    [1, 1,  24, 3, 2, 2,  0, 24,  24,  4,  4, 0, 2, 0, 1], #24->48(0, 0)->96  ->24(8,3)->24
    [2, 1,  32, 5, 1, 6,  6,  6,  32,  4,  4, 0, 2, 0, 1], #24->24(2,12)->144  ->32(16,2)->32
    [1, 1,  32, 5, 1, 6,  8,  8,  32,  4,  4, 0, 2, 0, 2], #32->32(2,16)->192 ->32(16,2)->32
    [1, 1,  64, 5, 1, 6,  8,  8,  48,  8,  8, 0, 2, 0, 2], #32->32(2,16)->192 ->48(12,4)->48
    [1, 1,  80, 5, 1, 6,  8,  8,  80,  8,  8, 0, 2, 0, 2], #48->48(3,16)->288 ->80(16,5)->80
    [1, 1,  80, 5, 1, 6, 10, 10,  80,  8,  8, 0, 2, 0, 2], #80->80(4,20)->480->80(20,4)->80
    [2, 1, 120, 5, 1, 6, 10, 10, 120, 10, 10, 0, 2, 0, 2], #80->80(4,20)->480->128(16,8)->128
    [1, 1, 120, 5, 1, 6, 12, 12, 120, 10, 10, 0, 2, 0, 2], #120->128(4,32)->720->128(32,4)->120
    [1, 1, 144, 3, 1, 6, 12, 12, 144, 12, 12, 0, 2, 0, 2], #120->128(4,32)->720->160(32,5)->144
    [1, 1, 864, 3, 1, 6, 12, 12,   0,  0,  0, 0, 2, 0, 2], #144->144(5,32)->864
]

def get_micronet_config(mode):
    return eval(mode+'_cfgs')

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


########################################################################
# sigmoid and tanh
########################################################################
# h_sigmoid (x: [-3 3], y: [0, h_max]]
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max / 6

    def forward(self, x):
        return self.relu(x + 3) * self.h_max

# h_tanh x: [-3, 3], y: [-h_max, h_max]
class h_tanh(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super(h_tanh, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3)*self.h_max / 3 - self.h_max


########################################################################
# wrap functions
########################################################################

def get_act_layer(inp, oup, mode='SE1', act_relu=True, act_max=2, act_bias=True, init_a=[1.0, 0.0], reduction=4, init_b=[0.0, 0.0], g=None, act='relu', expansion=True):
    layer = None
    if mode == 'SE1':
        layer = nn.Sequential(
            SELayer(inp, oup, reduction=reduction), 
            nn.ReLU6(inplace=True) if act_relu else nn.Sequential()
        )
    elif mode == 'SE0':
        layer = nn.Sequential(
            SELayer(inp, oup, reduction=reduction), 
        )
    elif mode == 'NA':
        layer = nn.ReLU6(inplace=True) if act_relu else nn.Sequential()
    elif mode == 'LeakyReLU':
        layer = nn.LeakyReLU(inplace=True) if act_relu else nn.Sequential()
    elif mode == 'RReLU':
        layer = nn.RReLU(inplace=True) if act_relu else nn.Sequential()
    elif mode == 'PReLU':
        layer = nn.PReLU() if act_relu else nn.Sequential()
    elif mode == 'DYShiftMax':
        layer = DYShiftMax(inp, oup, act_max=act_max, act_relu=act_relu, init_a=init_a, reduction=reduction, init_b=init_b, g=g, expansion=expansion)
    return layer

########################################################################
# dynamic activation layers (SE, DYShiftMax, etc)
########################################################################

class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.oup = oup
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # determine squeeze
        squeeze = get_squeeze_channels(inp, reduction)
        # print('reduction: {}, squeeze: {}/{}'.format(reduction, inp, squeeze))


        self.fc = nn.Sequential(
                nn.Linear(inp, squeeze),
                nn.ReLU(inplace=True),
                nn.Linear(squeeze, oup),
                h_sigmoid()
        )

    def forward(self, x):
        if isinstance(x, list):
            x_in = x[0]
            x_out = x[1]
        else:
            x_in = x
            x_out = x
        b, c, _, _ = x_in.size()
        y = self.avg_pool(x_in).view(b, c)
        y = self.fc(y).view(b, self.oup, 1, 1)
        return x_out * y

class DYShiftMax(nn.Module):
    def __init__(self, inp, oup, reduction=4, act_max=1.0, act_relu=True, init_a=[0.0, 0.0], init_b=[0.0, 0.0], relu_before_pool=False, g=None, expansion=False):
        super(DYShiftMax, self).__init__()
        self.oup = oup
        self.act_max = act_max * 2
        self.act_relu = act_relu
        self.avg_pool = nn.Sequential(
                nn.ReLU(inplace=True) if relu_before_pool == True else nn.Sequential(),
                nn.AdaptiveAvgPool2d(1)
            )

        self.exp = 4 if act_relu else 2
        self.init_a = init_a
        self.init_b = init_b

        # determine squeeze
        squeeze = _make_divisible(inp // reduction, 4)
        if squeeze < 4:
            squeeze = 4
        # print('reduction: {}, squeeze: {}/{}'.format(reduction, inp, squeeze))
        # print('init-a: {}, init-b: {}'.format(init_a, init_b))

        self.fc = nn.Sequential(
                nn.Linear(inp, squeeze),
                nn.ReLU(inplace=True),
                nn.Linear(squeeze, oup*self.exp),
                h_sigmoid()
        )
        if g is None:
            g = 1
        self.g = g[1]
        if self.g !=1  and expansion:
            self.g = inp // self.g
        # print('group shuffle: {}, divide group: {}'.format(self.g, expansion))
        self.gc = inp//self.g
        index=torch.Tensor(range(inp)).view(1,inp,1,1)
        index=index.view(1,self.g,self.gc,1,1)
        indexgs = torch.split(index, [1, self.g-1], dim=1)
        indexgs = torch.cat((indexgs[1], indexgs[0]), dim=1)
        indexs = torch.split(indexgs, [1, self.gc-1], dim=2)
        indexs = torch.cat((indexs[1], indexs[0]), dim=2)
        self.index = indexs.view(inp).type(torch.LongTensor)
        self.expansion = expansion

    def forward(self, x):
        x_in = x
        x_out = x

        b, c, _, _ = x_in.size()
        y = self.avg_pool(x_in).view(b, c)
        y = self.fc(y).view(b, self.oup*self.exp, 1, 1)
        y = (y-0.5) * self.act_max

        n2, c2, h2, w2 = x_out.size()
        x2 = x_out[:,self.index,:,:]

        if self.exp == 4:
            a1, b1, a2, b2 = torch.split(y, self.oup, dim=1)

            a1 = a1 + self.init_a[0]
            a2 = a2 + self.init_a[1]

            b1 = b1 + self.init_b[0]
            b2 = b2 + self.init_b[1]

            z1 = x_out * a1 + x2 * b1
            z2 = x_out * a2 + x2 * b2

            out = torch.max(z1, z2)

        elif self.exp == 2:
            a1, b1 = torch.split(y, self.oup, dim=1)
            a1 = a1 + self.init_a[0]
            b1 = b1 + self.init_b[0]
            out = x_out * a1 + x2 * b1

        return out

def get_squeeze_channels(inp, reduction):
    if reduction == 4:
        squeeze = inp // reduction
    else:
        squeeze = _make_divisible(inp // reduction, 4)
    return squeeze


TAU = 20
#####################################################################3
# part 1: functions
#####################################################################3

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

def conv_3x3_bn(inp, oup, stride, dilation=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False, dilation=dilation),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def gcd(a, b):
    a, b = (a, b) if a >= b else (b, a)
    while b:
        a, b = b, a%b
    return a

#####################################################################3
# part 2: modules
#####################################################################3

class MaxGroupPooling(nn.Module):
    def __init__(self, channel_per_group=2):
        super(MaxGroupPooling, self).__init__()
        self.channel_per_group = channel_per_group

    def forward(self, x):
        if self.channel_per_group == 1:
            return x
        # max op
        b, c, h, w = x.size()

        # reshape
        y = x.view(b, c // self.channel_per_group, -1, h, w)
        out, _ = torch.max(y, dim=2)
        return out

class SwishLinear(nn.Module):
    def __init__(self, inp, oup):
        super(SwishLinear, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(inp, oup),
            nn.BatchNorm1d(oup),
            h_swish()
        )

    def forward(self, x):
        return self.linear(x)
    
class StemLayer(nn.Module):
    def __init__(self, inp, oup, stride, dilation=1, mode='default', groups=(4,4)):
        super(StemLayer, self).__init__()

        self.exp = 1 if mode == 'default' else 2
        g1, g2 = groups 
        if mode == 'default':
            self.stem = nn.Sequential(
                nn.Conv2d(inp, oup*self.exp, 3, stride, 1, bias=False, dilation=dilation),
                nn.BatchNorm2d(oup*self.exp),
                nn.ReLU6(inplace=True) if self.exp == 1 else MaxGroupPooling(self.exp)
            )
        elif mode == 'spatialsepsf':
            self.stem = nn.Sequential(
                SpatialSepConvSF(inp, groups, 3, stride),
                MaxGroupPooling(2) if g1*g2==2*oup else nn.ReLU6(inplace=True)
            )
        else: 
            exp = 6
            ch_per_group=2
            hidden_dim = inp*exp //ch_per_group
            self.stem = nn.Sequential(
                DepthExpandConv(inp, exp, kernel_size=3, stride=stride),
                MaxGroupPooling(ch_per_group),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False, groups=1),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )
           
    def forward(self, x):
        out = self.stem(x)    
        return out

class GroupConv(nn.Module):
    def __init__(self, inp, oup, groups=2):
        super(GroupConv, self).__init__()
        self.inp = inp
        self.oup = oup
        self.groups = groups
        # print ('inp: %d, oup:%d, g:%d' %(inp, oup, self.groups[0]))
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False, groups=self.groups[0]),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()

        channels_per_group = c // self.groups

        # reshape
        x = x.view(b, self.groups, channels_per_group, h, w)

        x = torch.transpose(x, 1, 2).contiguous()
        out = x.view(b, -1, h, w)

        return out

class ChannelShuffle2(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle2, self).__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()

        channels_per_group = c // self.groups

        # reshape
        x = x.view(b, self.groups, channels_per_group, h, w)

        x = torch.transpose(x, 1, 2).contiguous()
        out = x.view(b, -1, h, w)

        return out

######################################################################3
# part 3: new block
#####################################################################3

class SpatialSepConvSF(nn.Module):
    def __init__(self, inp, oups, kernel_size, stride):
        super(SpatialSepConvSF, self).__init__()

        oup1, oup2 = oups
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup1,
                (kernel_size, 1),
                (stride, 1),
                (kernel_size//2, 0),
                bias=False, groups=1
            ),
            nn.BatchNorm2d(oup1),
            nn.Conv2d(oup1, oup1*oup2,
                (1, kernel_size),
                (1, stride),
                (0, kernel_size//2),
                bias=False, groups=oup1
            ),
            nn.BatchNorm2d(oup1*oup2),
            ChannelShuffle(oup1),
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class DepthConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride):
        super(DepthConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, bias=False, groups=inp),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class DepthSpatialSepConv(nn.Module):
    def __init__(self, inp, expand, kernel_size, stride):
        super(DepthSpatialSepConv, self).__init__()

        exp1, exp2 = expand

        hidden_dim = inp*exp1
        oup = inp*exp1*exp2
        
        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp*exp1, 
                (kernel_size, 1), 
                (stride, 1), 
                (kernel_size//2, 0), 
                bias=False, groups=inp
            ),
            nn.BatchNorm2d(inp*exp1),
            nn.Conv2d(hidden_dim, oup,
                (1, kernel_size),
                (1, stride),
                (0, kernel_size//2),
                bias=False, groups=hidden_dim
            ),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        out = self.conv(x)
        return out
    
def get_pointwise_conv(mode, inp, oup, hiddendim, groups):

    if mode == 'group':
        return GroupConv(inp, oup, groups)
    elif mode == '1x1':
        return nn.Sequential(
                    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup)
                )
    else:
        return None

class DYMicroBlock(nn.Module):
    def __init__(self,
                 inp,
                 oup,
                 kernel_size=3,
                 stride=1,
                 ch_exp=(2, 2),
                 ch_per_group=4,
                 groups_1x1=(1, 1),
                 depthsep=True,
                 shuffle=False,
                 pointwise='fft',
                 *, activation_cfg: dict):
        super(DYMicroBlock, self).__init__()

        # print(activation_cfg.dy)

        self.identity = stride == 1 and inp == oup

        y1, y2, y3 = activation_cfg['dy']
        act = activation_cfg['module']
        act_max = activation_cfg['act_max']
        act_bias = activation_cfg['linearse_bias']
        act_reduction = activation_cfg['reduction'] * activation_cfg['ratio']
        init_a = activation_cfg['init_a']
        init_b = activation_cfg['init_b']
        init_ab3 = activation_cfg['init_a_block3']

        t1 = ch_exp
        gs1 = ch_per_group
        hidden_fft, g1, g2 = groups_1x1

        hidden_dim1 = inp * t1[0]
        hidden_dim2 = inp * t1[0] * t1[1]

        if gs1[0] == 0:
            self.layers = nn.Sequential(
                DepthSpatialSepConv(inp, t1, kernel_size, stride),
                get_act_layer(
                    hidden_dim2,
                    hidden_dim2,
                    mode=act,
                    act_max=act_max,
                    act_relu=True if y2 == 2 else False,
                    act_bias=act_bias,
                    init_a=init_a,
                    reduction=act_reduction,
                    init_b=init_b,
                    g = gs1,
                    expansion = False
                ) if y2 > 0 else nn.ReLU6(inplace=True),
                ChannelShuffle(gs1[1]) if shuffle else nn.Sequential(),
                ChannelShuffle2(hidden_dim2//2) if shuffle and y2 !=0 else nn.Sequential(),
                get_pointwise_conv(pointwise, hidden_dim2, oup, hidden_fft, (g1, g2)),
                get_act_layer(
                    oup,
                    oup,
                    mode=act,
                    act_max=act_max,
                    act_relu=False,
                    act_bias=act_bias,
                    init_a=[init_ab3[0], 0.0],
                    reduction=act_reduction//2,
                    init_b=[init_ab3[1], 0.0],
                    g = (g1, g2),
                    expansion = False
                ) if y3 > 0 else nn.Sequential(),
                ChannelShuffle(g2) if shuffle else nn.Sequential(),
                ChannelShuffle2(oup//2) if shuffle and oup%2 == 0  and y3!=0 else nn.Sequential(),
            )
        elif g2 == 0:
            self.layers = nn.Sequential(
                get_pointwise_conv(pointwise, inp, hidden_dim2, hidden_dim1, gs1),
                get_act_layer(
                    hidden_dim2,
                    hidden_dim2,
                    mode=act,
                    act_max=act_max,
                    act_relu=False,
                    act_bias=act_bias,
                    init_a=[init_ab3[0], 0.0],
                    reduction=act_reduction,
                    init_b=[init_ab3[1], 0.0],
                    g = gs1,
                    expansion = False
                ) if y3 > 0 else nn.Sequential(),

            )

        else:
            self.layers = nn.Sequential(
                get_pointwise_conv(pointwise, inp, hidden_dim2, hidden_dim1, gs1),
                get_act_layer(
                    hidden_dim2,
                    hidden_dim2,
                    mode=act,
                    act_max=act_max,
                    act_relu=True if y1 == 2 else False,
                    act_bias=act_bias,
                    init_a=init_a,
                    reduction=act_reduction,
                    init_b=init_b,
                    g = gs1,
                    expansion = False
                ) if y1 > 0 else nn.ReLU6(inplace=True),
                ChannelShuffle(gs1[1]) if shuffle else nn.Sequential(),
                DepthSpatialSepConv(hidden_dim2, (1, 1), kernel_size, stride) if depthsep else
                DepthConv(hidden_dim2, hidden_dim2, kernel_size, stride),
                nn.Sequential(),
                get_act_layer(
                    hidden_dim2,
                    hidden_dim2,
                    mode=act,
                    act_max=act_max,
                    act_relu=True if y2 == 2 else False,
                    act_bias=act_bias,
                    init_a=init_a,
                    reduction=act_reduction,
                    init_b=init_b,
                    g = gs1,
                    expansion = True
                ) if y2 > 0 else nn.ReLU6(inplace=True),
                ChannelShuffle2(hidden_dim2//4) if shuffle and y1!=0 and y2 !=0 else nn.Sequential() if y1==0 and y2==0 else ChannelShuffle2(hidden_dim2//2),
                get_pointwise_conv(pointwise, hidden_dim2, oup, hidden_fft, (g1, g2)), #FFTConv
                get_act_layer(
                    oup,
                    oup,
                    mode=act,
                    act_max=act_max,
                    act_relu=False,
                    act_bias=act_bias,
                    init_a=[init_ab3[0], 0.0],
                    reduction=act_reduction//2 if oup < hidden_dim2 else act_reduction,
                    init_b=[init_ab3[1], 0.0],
                    g = (g1, g2),
                    expansion = False
                ) if y3 > 0 else nn.Sequential(),
                ChannelShuffle(g2) if shuffle else nn.Sequential(),
                ChannelShuffle2(oup//2) if shuffle and y3!=0 else nn.Sequential(),
            )

    def forward(self, x):
        identity = x
        out = self.layers(x)

        if self.identity:
            out = out + identity

        return out

###########################################################################

class MicroNet(nn.Module):
    def __init__(self,
                 cfg: dict,
                 input_size=224,
                 num_classes=1000,
                 teacher=False):
        super(MicroNet, self).__init__()
        # setting of inverted residual blocks

        mode = cfg['net_config']
        self.cfgs = get_micronet_config(mode)

        block = eval(cfg['block'])
        stem_mode = cfg['stem_mode']
        stem_ch = cfg['stem_ch']
        stem_dilation = cfg['stem_dilation']
        stem_groups = cfg['stem_groups']
        out_ch = cfg['out_ch']
        depthsep = cfg['depthsep']
        shuffle = cfg['shuffle']
        pointwise = cfg['pointwise']
        dropout_rate = cfg['dropout']

        act_max = cfg['activation']['act_max']
        act_bias = cfg['activation']['linearse_bias']
        activation_cfg= cfg['activation']

        # building first layer
        assert input_size % 32 == 0
        input_channel = stem_ch
        layers = [StemLayer(
                    3, input_channel,
                    stride=2, 
                    dilation=stem_dilation, 
                    mode=stem_mode,
                    groups=stem_groups
                )]

        for idx, val in enumerate(self.cfgs):
            s, n, c, ks, c1, c2, g1, g2, c3, g3, g4, y1, y2, y3, r = val

            t1 = (c1, c2)
            gs1 = (g1, g2)
            gs2 = (c3, g3, g4)
            activation_cfg['dy'] = [y1, y2, y3]
            activation_cfg['ratio'] = r

            output_channel = c
            layers.append(block(input_channel, output_channel,
                kernel_size=ks, 
                stride=s, 
                ch_exp=t1, 
                ch_per_group=gs1, 
                groups_1x1=gs2,
                depthsep = depthsep,
                shuffle = shuffle,
                pointwise = pointwise,
                activation_cfg=activation_cfg,
            ))
            input_channel = output_channel
            for i in range(1, n):
                layers.append(block(input_channel, output_channel, 
                    kernel_size=ks, 
                    stride=1, 
                    ch_exp=t1, 
                    ch_per_group=gs1, 
                    groups_1x1=gs2,
                    depthsep = depthsep,
                    shuffle = shuffle,
                    pointwise = pointwise,
                    activation_cfg=activation_cfg,
                ))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)


        self.avgpool = nn.Sequential(
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            h_swish()
        ) 

        # building last several layers
        output_channel = out_ch
         
        self.classifier = nn.Sequential(
            SwishLinear(input_channel, output_channel),
            nn.Dropout(dropout_rate),
            SwishLinear(output_channel, num_classes)
        )
        self._initialize_weights()
           
    def _forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

def _load_pretrained(model_name, model, progress):
    if model_name not in _MODEL_URLS or _MODEL_URLS[model_name] is None:
        raise ValueError(
            "No checkpoint is available for model type {}".format(model_name))
    checkpoint_url = _MODEL_URLS[model_name]
    mmcv.runner.load_state_dict(
        model, load_state_dict_from_url(checkpoint_url, progress=progress))


# def micronet(mode, pretrained=False, progress=True, **kwargs):
#     model = MicroNet(**kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(
#             model_urls[mode], progress=progress)
#         model.load_state_dict(state_dict)
#     return model


@BACKBONES.register_module()
class MicroNetBackbone(MicroNet):
    """
    
    """
    def __init__(self,
                 mode: str,
                 out_indices:tuple,
                 cfg: dict=None,
                 input_size=224,
                 num_classes=1000,
                 **kwargs) -> None:
        """
        input_size: should be input_size % 32 == 0
        cfg will overwrite mode (some or all it`s up to you)
        """

        micronet_cfg = _MODEL_CFG.get(mode.lower(), None)

        assert micronet_cfg is not None or cfg is not None, f"""
            dont have this mode support: {mode}
        """
        
        assert out_indices.__len__() != 0, """
          out_indices len should > 0
        """
        
        self.out_indices = out_indices

        if cfg is not None:
            tuple(map(self.replace(micronet_cfg), cfg.items()))
            
        assert max(out_indices) < len(get_micronet_config(micronet_cfg['net_config'])),f"""
          mode: {mode} layer_len is: {len(get_micronet_config(micronet_cfg['net_config']))}
          but out_indices have :{max(out_indices)} index, out of range
          index start is 0
        """

        super(MicroNetBackbone, self).__init__(
            cfg=micronet_cfg, input_size=input_size, num_classes=num_classes, **kwargs)

    @staticmethod
    def replace(cfg_dcit):

        def temp_func(key_value: tuple):
            cfg_dcit[key_value[0]] = key_value[1]
        
        return temp_func

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            _load_pretrained(pretrained, self, progress=True)
        elif pretrained is None:
            self._initialize_weights()
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.features[0](x)
        
        out = list()
        for i, layer in enumerate(self.features[1:]):
          x = layer(x)
          if i in self.out_indices:
            out.append(x)
          
        # print(len(self.features))
        return tuple(out)
    

if __name__ == '__main__':
    model = MicroNetBackbone(mode="m2", out_indices=(5, 6, 7), cfg=None)
    dummy_input = Variable(torch.randn(1, 3, 416, 416))
    print(model)
    outs = model(dummy_input)
    for out in outs:
        print(out.size())