import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.layers import denseBlock2, convBlock, unetUp, outconv

class Net(nn.Module):
    """Some Information about Net"""
    def __init__(self, in_channels=3, n_classes=2, feature_scale=2, is_deconv=True, is_batchnorm=True):
        super(Net, self).__init__()
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.lowconv = nn.Conv2d(in_channels, 32, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.dense1 = denseBlock2(32, 128, 32, 3)
        self.maxpool1 = nn.MaxPool2d(2)
        self.dense2 = denseBlock2(128, 256, 32, 3)
        self.maxpool2 = nn.MaxPool2d(2)
        self.dense3 = denseBlock2(224, 352, 32, 4)
        self.maxpool3 = nn.MaxPool2d(2)
        self.dense4 = denseBlock2(352, 576, 32, 6)
        self.maxpool4 = nn.MaxPool2d(2)
        self.trasit1 = convBlock(128,68)
        self.trasit2 = convBlock(224,136)
        self.trasit3 = convBlock(352,272)
        self.bottleneck = nn.Conv2d(544, 1088, 3, 1, 1)
        self.up_cat1 = unetUp(1088, 544, self.is_deconv)
        self.up_cat2 = unetUp(544, 272, self.is_deconv)
        self.up_cat3 = unetUp(272, 136, self.is_deconv)
        self.up_cat4 = unetUp(136, 68, self.is_deconv)
        self.outconv = outconv(68, n_classes)

    def forward(self, x):
        c1 = self.lowconv(x)
        c1 = self.relu(c1)
        # print(c1.shape)
        d1 = self.dense1(c1)
        d1_ = self.trasit1(d1)
        # print(d1.shape)
        m1 = self.maxpool1(d1)
        # print(m1.shape)
        d2 = self.dense2(m1)
        d2_ = self.trasit2(d2)
        # print(d2.shape)
        m2 = self.maxpool2(d2)
        # print(m2.shape)
        d3 = self.dense3(m2)
        d3_ = self.trasit3(d3)
        # print(d3.shape)
        m3 = self.maxpool3(d3)
        # print(m3.shape)
        d4 = self.dense4(m3)
        # print(d4.shape)
        m4 = self.maxpool4(d4)
        # print(m4.shape)
        bn = self.bottleneck(m4)
        # print(bn.shape)
        up1 = self.up_cat1(bn, d4)
        # print(up1.shape)
        up2 = self.up_cat2(up1, d3_)
        # print(up2.shape)
        up3 = self.up_cat3(up2, d2_)
        # print(up3.shape)
        up4 = self.up_cat4(up3, d1_)
        # print(up4.shape)

        out = self.outconv(up4)
        # print(out.shape)

        return out


class Net2(nn.Module):
    """Some Information about Net"""
    def __init__(self, in_channels=1, n_classes=2, feature_scale=2, is_deconv=True, is_batchnorm=True):
        super(Net2, self).__init__()
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.lowconv = nn.Conv2d(in_channels, 32, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.dense1 = denseBlock2(32, 64, 32, 1)
        self.maxpool1 = nn.MaxPool2d(2)
        self.dense2 = denseBlock2(64, 128, 32, 2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.dense3 = denseBlock2(128, 256, 32, 4)
        self.maxpool3 = nn.MaxPool2d(2)
        self.dense4 = denseBlock2(256, 512, 32, 8)
        self.maxpool4 = nn.MaxPool2d(2)
        # self.trasit1 = convBlock(128,68)
        # self.trasit2 = convBlock(224,136)
        # self.trasit3 = convBlock(352,272)
        self.bottleneck = nn.Conv2d(512, 1024, 3, 1, 1)
        self.up_cat1 = unetUp(1024, 512, self.is_deconv)
        self.up_cat2 = unetUp(512, 256, self.is_deconv)
        self.up_cat3 = unetUp(256, 128, self.is_deconv)
        self.up_cat4 = unetUp(128, 64, self.is_deconv)
        self.outconv = outconv(64, n_classes)

    def forward(self, x):
        c1 = self.lowconv(x)
        c1 = self.relu(c1)
        # print(c1.shape)
        d1 = self.dense1(c1)
        # d1_ = self.trasit1(d1)
        # print(d1.shape)
        m1 = self.maxpool1(d1)
        # print(m1.shape)
        d2 = self.dense2(m1)
        # d2_ = self.trasit2(d2)
        # print(d2.shape)
        m2 = self.maxpool2(d2)
        # print(m2.shape)
        d3 = self.dense3(m2)
        # d3_ = self.trasit3(d3)
        # print(d3.shape)
        m3 = self.maxpool3(d3)
        # print(m3.shape)
        d4 = self.dense4(m3)
        # print(d4.shape)
        m4 = self.maxpool4(d4)
        # print(m4.shape)
        bn = self.bottleneck(m4)
        # print(bn.shape)
        up1 = self.up_cat1(bn, d4)
        # print(up1.shape)
        up2 = self.up_cat2(up1, d3)
        # print(up2.shape)
        up3 = self.up_cat3(up2, d2)
        # print(up3.shape)
        up4 = self.up_cat4(up3, d1)
        # print(up4.shape)

        out = self.outconv(up4)
        # print(out.shape)

        return out

class Net3(nn.Module):
    """Some Information about Net"""
    def __init__(self, in_channels=3, n_classes=1, feature_scale=2, is_deconv=False, is_batchnorm=True):
        super(Net3, self).__init__()
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.lowconv = nn.Conv2d(in_channels, 32, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.dense1 = denseBlock2(32, 64, 8, 4)
        self.maxpool1 = nn.MaxPool2d(2)
        self.dense2 = denseBlock2(64, 96, 8, 4)
        self.maxpool2 = nn.MaxPool2d(2)
        self.dense3 = denseBlock2(96, 128, 8, 4)
        self.maxpool3 = nn.MaxPool2d(2)
        self.dense4 = denseBlock2(128, 160, 8, 4)
        self.maxpool4 = nn.MaxPool2d(2)
        self.trasit1 = nn.Conv2d(64,40, kernel_size=1, padding=0, stride=1)
        self.trasit2 = nn.Conv2d(96,80, kernel_size=1, padding=0, stride=1)
        self.trasit3 = nn.Conv2d(128,160, kernel_size=1, padding=0, stride=1)
        self.trasit4 = nn.Conv2d(160,320, kernel_size=1, padding=0, stride=1)
        self.bottleneck = nn.Conv2d(160, 640, 3, 1, 1)
        self.up_cat1 = unetUp(640, 320, self.is_deconv)
        self.up_cat2 = unetUp(320, 160, self.is_deconv)
        self.up_cat3 = unetUp(160, 80, self.is_deconv)
        self.up_cat4 = unetUp(80, 40, self.is_deconv)
        self.outconv = outconv(40, n_classes)

    def forward(self, x):
        c1 = self.lowconv(x)
        c1 = self.relu(c1)
        # print(c1.shape)
        d1 = self.dense1(c1)
        d1_ = self.trasit1(d1)
        # print(d1.shape)
        m1 = self.maxpool1(d1)
        # print(m1.shape)
        d2 = self.dense2(m1)
        d2_ = self.trasit2(d2)
        # print(d2.shape)
        m2 = self.maxpool2(d2)
        # print(m2.shape)
        d3 = self.dense3(m2)
        d3_ = self.trasit3(d3)
        # print(d3.shape)
        m3 = self.maxpool3(d3)
        # print(m3.shape)
        d4 = self.dense4(m3)
        d4_ = self.trasit4(d4)
        # print(d4_.shape)
        m4 = self.maxpool4(d4)
        # print(m4.shape)
        bn = self.bottleneck(m4)
        # print(bn.shape)
        up1 = self.up_cat1(bn, d4_)
        # print(up1.shape)
        up2 = self.up_cat2(up1, d3_)
        # print(up2.shape)
        up3 = self.up_cat3(up2, d2_)
        # print(up3.shape)
        up4 = self.up_cat4(up3, d1_)
        # print(up4.shape)

        out = self.outconv(up4)
        # print(out.shape)

        return out