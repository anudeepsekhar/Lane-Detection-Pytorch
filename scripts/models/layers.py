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
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        # self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//2, 1),
            nn.BatchNorm2d(in_ch//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch//2, out_ch, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class convBlock(nn.Module):
    """Some Information about convBlock"""
    def __init__(self, in_channels, out_channels, ks=3, stride=1, padding=1, inter=False):
        super(convBlock, self).__init__()
        self.inter = inter

        if self.inter: 
            self.inter_channel = out_channels*4
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.relu1 = nn.ReLU()
            self.conv1 = nn.Conv2d(in_channels, self.inter_channel, kernel_size=1, stride=1, padding=0)
            self.bn3 = nn.BatchNorm2d(self.inter_channel)
            self.relu3 = nn.ReLU()
            self.conv3 = nn.Conv2d(self.inter_channel, out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.bn = nn.BatchNorm2d(in_channels)
            self.relu = nn.ReLU()
            self.conv = nn.Conv2d(in_channels, out_channels, ks, stride, padding)
        
    def forward(self, x):
        if self.inter:    
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.conv1(x)
            x = self.bn3(x)
            x = self.relu3(x)
            x = self.conv3(x)
        else:
            x = self.bn(x)
            x = self.relu(x)
            x = self.conv(x)
        return x

class denseBlock2(nn.Module):
    """Some Information about denseBlock2"""
    def __init__(self, in_channels, out_channels, growth_rate, num_convs):
        super(denseBlock2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.growth_rate = growth_rate
        self.net = nn.Sequential()
        for i in range(num_convs):
            self.net.add_module(f'conv{i}', convBlock(self.in_channels, self.growth_rate, inter=True))
            self.in_channels = self.in_channels + self.growth_rate
        self.out_conv = convBlock(self.in_channels, self.out_channels, ks=3)

    def forward(self, x):
        for blk in self.net.children():
            y = blk(x)
            x = torch.cat([x, y], 1)
            # print('dense: ', x.shape)
            # self.out_conv(x)
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


class vggUpconv(nn.Module):
    """Some Information about vggUpconv"""
    def __init__(self, in_ch, out_ch, upsample= True):
        super(vggUpconv, self).__init__()
        if upsample:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            self.upsample = nn.Conv2d(in_ch, in_ch, 3, 1, 1) 
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)


    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x1 = self.conv1(x1)
        sum = x1 + x2

        return sum