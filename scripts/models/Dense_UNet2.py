import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.layers import denseBlock2, unetUp, convBlock, unetConv2

class Dense_UNet2(nn.Module):
    """Some Information about Dense_UNet2"""
    def __init__(self, in_channels=3, n_classes=2, feature_scale=2, is_deconv=True, is_batchnorm=True):
        super(Dense_UNet2, self).__init__()
        self.lowconv = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=7, padding=3, bias=False)
        self.relu = nn.ReLU()
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm

        filters = [64, 128, 256, 512, 1024]

        self.dense_block1 = denseBlock2(32, filters[0], 3)
        self.dense_block2 = denseBlock2(filters[0], filters[1], 3)
        self.dense_block3 = denseBlock2(filters[1], filters[2], 3)
        self.dense_block4 = denseBlock2(filters[2], filters[3], 3)

        self.maxpool = nn.MaxPool2d(kernel_size=2) 

        self.bottleneck = unetConv2(filters[3], filters[4], self.is_batchnorm)

        #upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        # final conv (without any concat)
        self.out_conv = nn.Conv2d(filters[0], n_classes, 1)


    def forward(self, x):
        c1 = self.lowconv(x) #128x128x32

        d1 = self.dense_block1(c1) #128x128x64
        m1 = self.maxpool(d1) #64x64x64

        d2 = self.dense_block2(m1) #64x64x128
        m2 = self.maxpool(d2) #32x32x128

        d3 = self.dense_block3(m2) #32x32x256
        m3 = self.maxpool(d3) #16x16x256

        d4 = self.dense_block4(m3) #16x16x512
        m4 = self.maxpool(d4) #8x8x512

        bn = self.bottleneck(m4)
        # print(bn.size)

        up4 = self.up_concat4(bn, d4)
        up3 = self.up_concat3(up4, d3)
        up2 = self.up_concat2(up3, d2)
        up1 = self.up_concat1(up2, d1)

        out = self.out_conv(up1)

        return out