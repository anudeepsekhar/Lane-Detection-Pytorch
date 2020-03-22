import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.layers import unetConv2, unetUp
from models.utils import init_weights

class UNet(nn.Module):
    """Some Information about UNet"""
    def __init__(self, in_channels=1, n_classes=2, feature_scale=2, is_deconv=True, is_batchnorm=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self. is_batchnorm = is_batchnorm

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x/self.feature_scale) for x in filters]

        #downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = unetConv2(self.in_channels,filters[0],self.is_batchnorm)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.bottleneck = unetConv2(filters[3], filters[4], self.is_batchnorm)

        #upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        # final conv (without any concat)
        self.out_conv = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')



    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool(conv4)

        bn = self.bottleneck(maxpool4)
        # print(bn.size)

        up4 = self.up_concat4(bn, conv4)
        up3 = self.up_concat3(up4, conv3)
        up2 = self.up_concat2(up3, conv2)
        up1 = self.up_concat1(up2, conv1)

        out = self.out_conv(up1)

        return out