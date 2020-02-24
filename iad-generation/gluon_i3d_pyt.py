import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict

class KitModel(nn.Module):

    
    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)

        self.conv0_fwd = self.__conv(3, name='conv0_fwd', in_channels=3, out_channels=64, kernel_size=(5L, 7L, 7L), stride=(2L, 2L, 2L), groups=1, bias=False)
        self.batchnorm0_fwd = self.__batch_normalization(3, 'batchnorm0_fwd', num_features=64, eps=9.99999974738e-06, momentum=0.899999976158)
        self.layer1_0_conv0_fwd = self.__conv(3, name='layer1_0_conv0_fwd', in_channels=64, out_channels=64, kernel_size=(3L, 1L, 1L), stride=(1L, 1L, 1L), groups=1, bias=False)
        self.layer1_downsample_conv0_fwd = self.__conv(3, name='layer1_downsample_conv0_fwd', in_channels=64, out_channels=256, kernel_size=(1L, 1L, 1L), stride=(1L, 1L, 1L), groups=1, bias=False)
        self.layer1_0_batchnorm0_fwd = self.__batch_normalization(3, 'layer1_0_batchnorm0_fwd', num_features=64, eps=9.99999974738e-06, momentum=0.899999976158)
        self.layer1_downsample_batchnorm0_fwd = self.__batch_normalization(3, 'layer1_downsample_batchnorm0_fwd', num_features=256, eps=9.99999974738e-06, momentum=0.899999976158)

    def forward(self, x):
        conv0_fwd_pad   = F.pad(x, (3L, 3L, 3L, 3L, 2L, 2L))
        conv0_fwd       = self.conv0_fwd(conv0_fwd_pad)
        batchnorm0_fwd  = self.batchnorm0_fwd(conv0_fwd)
        relu0_fwd       = F.relu(batchnorm0_fwd)
        pool0_fwd_pad   = F.pad(relu0_fwd, (1L, 1L, 1L, 1L, 0L, 0L), value=float('-inf'))
        pool0_fwd, pool0_fwd_idx = F.max_pool3d(pool0_fwd_pad, kernel_size=(1L, 3L, 3L), stride=(2L, 2L, 2L), padding=0, ceil_mode=False, return_indices=True)
        layer1_0_conv0_fwd_pad = F.pad(pool0_fwd, (0L, 0L, 0L, 0L, 1L, 1L))
        layer1_0_conv0_fwd = self.layer1_0_conv0_fwd(layer1_0_conv0_fwd_pad)
        layer1_downsample_conv0_fwd = self.layer1_downsample_conv0_fwd(pool0_fwd)
        layer1_0_batchnorm0_fwd = self.layer1_0_batchnorm0_fwd(layer1_0_conv0_fwd)
        layer1_downsample_batchnorm0_fwd = self.layer1_downsample_batchnorm0_fwd(layer1_downsample_conv0_fwd)
        return layer1_downsample_batchnorm0_fwd, layer1_0_batchnorm0_fwd


    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if   dim == 0 or dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()

        if 'scale' in __weights_dict[name]:
            layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['scale']))
        else:
            layer.weight.data.fill_(1)

        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        else:
            layer.bias.data.fill_(0)

        layer.state_dict()['running_mean'].copy_(torch.from_numpy(__weights_dict[name]['mean']))
        layer.state_dict()['running_var'].copy_(torch.from_numpy(__weights_dict[name]['var']))
        return layer

    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer

