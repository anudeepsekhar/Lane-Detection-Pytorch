import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.utils import init_weights

class unetConv2(nn.Module):
    """Some Information about unetConv2"""
    def __init__(self, in_channels, out_channels, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding

        if is_batchnorm:
            for i in range(1,self.n+1):
                conv = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        ks,
                        stride,
                        padding
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
                setattr(self, f'conv{i}',conv)
                in_channels = out_channels

        else:
            for i in range(1,self.n+1):
                conv = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        ks,
                        stride,
                        padding),
                    nn.ReLU(inplace=True)
                )
                setattr(self, f'conv{i}',conv)
                in_channels = out_channels

        for m in self.children():
            init_weights(m, init_type='kaiming')
            

    def forward(self, x):

        for i in range(1,self.n+1):
            conv = getattr(self,f'conv{i}')
            x = conv(x)
        return x


class unetUp(nn.Module):
    """Some Information about unetUp"""
    def __init__(self, in_channels, out_channels, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_channels + (n_concat-2)*out_channels, out_channels, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)

        else:
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )

        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, high_channels, *low_channels):
        out = self.up(high_channels)
        # print(out.shape)
        # print(low_channels[0].shape)
        for ch in low_channels:
            out = torch.cat([out, ch], 1)
        x = self.conv(out) 

        return x

class convBlock(nn.Module):
    """Some Information about convBlock"""
    def __init__(self, in_channels, out_channels, ks=3, stride=1, padding=1):
        super(convBlock, self).__init__()
        self.blk = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ks, stride, padding),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels)

        )

    def forward(self, x):
        self.blk(x)
        return x

class denseBlock2(nn.Module):
    """Some Information about denseBlock2"""
    def __init__(self, in_channels, out_channels, num_convs):
        super(denseBlock2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.net = nn.Sequential()
        for i in range(num_convs):
            self.net.add_module(f'conv{i}', convBlock(self.in_channels, self.out_channels))
            self.in_channels = self.out_channels + self.in_channels

    def forward(self, x):
        for blk in self.net.children():
            y = blk(x)
            x = torch.cat([x, y], 1)

        return x


class denseBlock(nn.Module):
    """Some Information about denseBlock"""
    def __init__(self, in_channels, growth_rate=32):
        super(denseBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(num_features=in_channels)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=growth_rate, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=1*growth_rate, out_channels=growth_rate, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=2*growth_rate, out_channels=growth_rate, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=3*growth_rate, out_channels=growth_rate, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=4*growth_rate, out_channels=growth_rate, kernel_size=3, stride=1, padding=1)
        

    def forward(self, x):
        bn = self.bn(x)
        conv1 = self.relu(self.conv1(bn))
        conv2 = self.relu(self.conv2(conv1))
        c2_dense = self.relu(torch.cat([conv1,conv2],1))
        conv3 = self.relu(self.conv3(c2_dense))
        c3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))
        conv4 = self.relu(self.conv4(c3_dense))
        c4_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4], 1))
        conv5 = self.relu(self.conv5(c4_dense))
        c5_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))
        
        return c5_dense

class transitionBlock(nn.Module):
    """Some Information about transitionBlock"""
    def __init__(self, in_channels, out_channels):
        super(transitionBlock, self).__init__()
        self.conv = unetConv2(in_channels, out_channels, True, n=2, ks=3)
        self.avgpool = nn.AvgPool2d(2,stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)

        return x