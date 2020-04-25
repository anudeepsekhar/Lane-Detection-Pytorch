import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.layers import unetConv2, unetUp
from models.utils import init_weights

class Linear_block(nn.Module):
    """Some Information about Linear_block"""
    def __init__(self, w, h, n_points):
        super(Linear_block, self).__init__()
        in_features = w*h
        self.block = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 2*n_points))

    def forward(self, x):
        flat = torch.flatten(x,1)
        out = self.block(flat)
        return out

class Output(nn.Module):
    def __init__(self, in_size, out_size):
        super(Output, self).__init__()
        self.conv = bottleneck(in_size, out_size, acti=False)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        return outputs

class bottleneck(nn.Module):
    """Some Information about bottleneck"""
    def __init__(self, in_channels, out_channels, acti=True):
        super(bottleneck, self).__init__()
        self.acti = acti
        temp_channels = int(in_channels/4)
        if in_channels<4:
            temp_channels = in_channels

        self.conv1 = Conv2d_BatchNorm_Relu(in_channels, temp_channels, 1, 0, 1)
        self.conv2 = Conv2d_BatchNorm_Relu(temp_channels, temp_channels, 3, 1, 1)
        self.conv3 = Conv2d_BatchNorm_Relu(temp_channels, out_channels, 1, 0, 1) 

        self.residual = Conv2d_BatchNorm_Relu(in_channels, out_channels, 1, 0, 1)

    def forward(self, x):
        re = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if not self.acti:
            return out

        re = self.residual(x)
        out = out + re

        return out

class Conv2d_BatchNorm_Relu(nn.Module):
    """Some Information about Conv2d_BatchNorm_Relu"""
    def __init__(self, in_channels, n_filters, k_size, padding, stride, bias=True, acti=True):
        super(Conv2d_BatchNorm_Relu, self).__init__()
        if acti:
            self.cbr_unit = nn.Sequential(
                nn.Conv2d(in_channels, n_filters, k_size, stride=stride, padding=padding, bias= bias),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(inplace=True))
        else:
            self.cbr_unit = nn.Conv2d(in_channels, n_filters, k_size, stride=stride, padding=padding, bias= bias)


    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

class UNetPoints(nn.Module):
    """Some Information about UNet"""
    def __init__(self, in_channels=3, n_classes=1, feature_scale=2, is_deconv=True, is_batchnorm=True):
        super(UNetPoints, self).__init__()
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
        self.line_branch = Output(512, 1) 
        self.linear = Linear_block(16,16,30)

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
        # print(bn.size())

        line_brach = self.line_branch(bn)
        points = self.linear(line_brach)


        up4 = self.up_concat4(bn, conv4)
        up3 = self.up_concat3(up4, conv3)
        up2 = self.up_concat2(up3, conv2)
        up1 = self.up_concat1(up2, conv1)

        out = self.out_conv(up1)

        return out, points