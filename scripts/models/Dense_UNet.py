import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.layers import unetConv2, unetUp, transitionBlock, denseBlock
from models.utils import init_weights

class Dens_UNet(nn.Module):
    """Some Information about Dens_UNet"""
    def __init__(self, in_channels=3, n_classes=2, feature_scale=2, is_deconv=True, is_batchnorm=True):
        super(Dens_UNet, self).__init__()
        self.lowconv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, padding=3, bias=False)
        self.relu = nn.ReLU()
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm

        filters = [64, 128, 256, 512, 1024]
        # filters = [int(x/self.feature_scale) for x in filters]

        # Make Dense Blocks 
        self.denseblock1 = self._make_dense_block(denseBlock, filters[0]) 
        self.denseblock2 = self._make_dense_block(denseBlock, filters[1])
        self.denseblock3 = self._make_dense_block(denseBlock, filters[2])
        self.denseblock4 = self._make_dense_block(denseBlock, filters[3])
        # self.denseblock4 = self._make_dense_block(denseBlock, filters[4])
        
        # Make transition Layers 
        self.transitionLayer1 = self._make_transition_layer(transitionBlock, in_channels = 160, out_channels = filters[1]) 
        self.transitionLayer2 = self._make_transition_layer(transitionBlock, in_channels = 160, out_channels = filters[2]) 
        self.transitionLayer3 = self._make_transition_layer(transitionBlock, in_channels = 160, out_channels = filters[3])
        self.transitionLayer4 = self._make_transition_layer(transitionBlock, in_channels = 160, out_channels = filters[4])
        
        self.bottleneck = unetConv2(filters[4], filters[4], self.is_batchnorm)

        #upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        # final conv (without any concat)
        self.out_conv = nn.Conv2d(filters[0], n_classes, 1)
    
    def _make_dense_block(self, block, in_channels): 
        layers = [] 
        layers.append(block(in_channels)) 
        return nn.Sequential(*layers) 
    def _make_transition_layer(self, layer, in_channels, out_channels): 
        modules = [] 
        modules.append(layer(in_channels, out_channels)) 
        return nn.Sequential(*modules)

    def forward(self, x):
        c1 = self.relu(self.lowconv(x)) # out channels = 64
        # print(c1.shape)

        d1 = self.denseblock1(c1) # in_channels=64 and out_channels=160
        # print(d1.shape)
        t1 = self.transitionLayer1(d1) # in_channels=160 and out_channels=128
        # print(t1.shape)
        d2 = self.denseblock2(t1) # in_channels=128 and out_channels=160
        # print(d2.shape)
        t2 = self.transitionLayer2(d2) # in_channels=160 and out_channels=256
        # print(t2.shape)
        d3 = self.denseblock3(t2) # in_channels=256 and out_channels=160
        # print(d3.shape)
        t3 = self.transitionLayer3(d3) # in_channels=160 and out_channels=512
        # print(t3.shape)
        d4 = self.denseblock4(t3) # in_channels=256 and out_channels=160
        # print(d4.shape)
        t4 = self.transitionLayer4(d4) # in_channels=160 and out_channels=512
        # print(t4.shape)
        bn = self.bottleneck(t4)
        # print(bn.shape)

        up4 = self.up_concat4(bn, t3)
        up3 = self.up_concat3(up4, t2)
        up2 = self.up_concat2(up3, t1)
        up1 = self.up_concat1(up2, c1)

        out = self.out_conv(up1)

        return out