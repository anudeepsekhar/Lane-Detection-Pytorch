import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

######################################################################
##
## Convolution layer modules
##
######################################################################

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


class bottleneck_down(nn.Module):
    """Some Information about bottleneck_down"""
    def __init__(self, in_channels, out_channels):
        super(bottleneck_down, self).__init__()
        temp_channels = int(in_channels/4)
        if in_channels<4:
            temp_channels = in_channels
        
        self.conv1 = Conv2d_BatchNorm_Relu(in_channels, temp_channels, 1, 0, 1)
        self.conv2 = Conv2d_BatchNorm_Relu(temp_channels, temp_channels, 3, 1, 2)
        self.conv3 = Conv2d_BatchNorm_Relu(temp_channels, out_channels, 1, 0, 1) 

        self.residual = Conv2d_BatchNorm_Relu(in_channels, out_channels, 3, 1, 2)

    def forward(self, x):
        re = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        re = self.residual(x)

        out = out + re

        return out

class bottleneck_up(nn.Module):
    """Some Information about bottleneck_up"""
    def __init__(self, in_channels, out_channels):
        super(bottleneck_up, self).__init__()
        temp_channels = int(in_channels/4)
        if in_channels<4:
            temp_channels = in_channels

        self.conv1 = Conv2d_BatchNorm_Relu(in_channels, temp_channels, 1, 0, 1)
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(temp_channels, temp_channels, 3, 2, 1, 1),
            nn.BatchNorm2d(temp_channels),
            nn.ReLU()
        )
        self.conv3 = Conv2d_BatchNorm_Relu(temp_channels, out_channels, 1, 0 ,1)

        self.residual = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


    def forward(self, x):
        re = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        re = self.residual(re)

        out = out + re

        return out


class Output(nn.Module):
    def __init__(self, in_size, out_size):
        super(Output, self).__init__()
        self.conv = bottleneck(in_size, out_size, acti=False)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        return outputs


class hourglass_same(nn.Module):
    """Some Information about hourglass_same"""
    def __init__(self, in_channels, out_channels):
        super(hourglass_same, self).__init__()
        self.down1 = bottleneck_down(in_channels, out_channels)
        self.down2 = bottleneck_down(out_channels, out_channels)
        self.down3 = bottleneck_down(out_channels, out_channels)
        self.down4 = bottleneck_down(out_channels, out_channels)

        self.same1 = bottleneck(in_channels, out_channels)
        self.same2 = bottleneck(in_channels, out_channels)

        self.up2 = bottleneck_up(in_channels, out_channels)
        self.up3 = bottleneck_up(out_channels, out_channels)
        self.up4 = bottleneck_up(out_channels, out_channels)
        self.up5 = bottleneck_up(out_channels, out_channels)

        self.residual1 = bottleneck_down(in_channels, out_channels)
        self.residual2 = bottleneck_down(out_channels, out_channels)
        self.residual3 = bottleneck_down(out_channels, out_channels)
        self.residual4 = bottleneck_down(out_channels, out_channels)
        

    def forward(self, inputs):
        output1 = self.down1(inputs)
        output2 = self.down2(output1)
        output3 = self.down3(output2)
        output4 = self.down4(output3)

        outputs = self.same1(output4)
        outputs = self.same2(outputs)

        outputs = self.up2(outputs + self.residual4(output3))
        outputs = self.up3(outputs + self.residual3(output2))
        outputs = self.up4(outputs + self.residual2(output1))
        outputs = self.up5(outputs + self.residual1(inputs))
        return outputs


class resize_layer(nn.Module):
    """Some Information about resize_layer"""
    def __init__(self, in_channels, out_channels, acti=True):
        super(resize_layer, self).__init__()
        self.conv = Conv2d_BatchNorm_Relu(in_channels, int(out_channels/2), 7, 3, 2)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.re1 = bottleneck(int(out_channels/2), int(out_channels/2))
        self.re2 = bottleneck(int(out_channels/2), int(out_channels/2))
        self.re3 = bottleneck(int(out_channels/2), int(out_channels))

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.re1(outputs)
        outputs = self.maxpool(outputs)
        outputs = self.re2(outputs)
        outputs = self.maxpool(outputs)
        outputs = self.re3(outputs)

        return outputs


class resize_up(nn.Module):
    """Some Information about resize_up"""
    def __init__(self, in_channels, out_channels):
        super(resize_up, self).__init__()
        temp_channels = in_channels//4
        if in_channels<4:
            temp_channels = in_channels
        self.up1 = bottleneck_up(in_channels, temp_channels)
        self.up2 = bottleneck_up(temp_channels,temp_channels//2)
        self.up3 = bottleneck_up(temp_channels//2, temp_channels//4)
        self.out_conv = Output(temp_channels//4, out_channels)


    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        out = self.out_conv(x)

        return out


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


class hourglass_block(nn.Module):
    """Some Information about hourglass_block"""
    def __init__(self, in_channels, out_channels, acti=True, input_re=True):
        super(hourglass_block, self).__init__()
        self.conv = Conv2d_BatchNorm_Relu(in_channels, out_channels//2, 7, 3, 2)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.re1 = bottleneck(out_channels//2, out_channels//2)
        self.re2 = bottleneck(out_channels//2, out_channels//2)
        self.re3 = bottleneck(out_channels//2, out_channels)
        self.re4 = bottleneck(out_channels, out_channels)
        self.re5 = bottleneck_down(out_channels, out_channels)

        self.layer1 = hourglass_same(out_channels, out_channels)
        self.line_branch = Output(out_channels, 1) 
        self.out_line = Output(out_channels//2, 1) 
        self.linear = Linear_block(32,32,30)

        self.up2 = bottleneck_up(out_channels, out_channels)
        self.up3 = bottleneck_up(out_channels, out_channels//2)
        self.up4 = bottleneck_up(out_channels//2, out_channels//2)
        self.up5 = bottleneck_up(out_channels//2, out_channels//2)

        self.residual1 = bottleneck_down(in_channels, out_channels//2)
        self.residual2 = bottleneck_down(out_channels//2, out_channels//2)
        self.residual3 = bottleneck_down(out_channels//2, out_channels)
        self.residual4 = bottleneck_down(out_channels, out_channels)
             

        self.out = Output(out_channels, 1)

        self.input_re =input_re


    def forward(self, inputs):
        output1 = self.conv(inputs)
        output1 = self.re1(output1)
        # print('out1: ',output1.size())#128x128x64
        output2 = self.maxpool(output1)
        output2 = self.re2(output2)
        # print('out2: ',output2.size())#64x64
        output3 = self.maxpool(output2)
        output3 = self.re3(output3)
        # print('out3: ',output3.size())#32x32x128

        output4 = self.layer1(output3)
        output4 = self.re4(output4)#32x32x128
        line_branch = self.line_branch(output4)#32x32x1
        pts = self.linear(line_branch)#1x60
        if self.input_re:
            outputs = output4 + output3
        else:
            outputs = output4 + out_line
        outputs = self.re5(outputs)
        # print('outputs: ', outputs.size())
        # print('res4: ', self.residual4(output3).size())
        # print('res3: ', self.residual3(output2).size())
        # print('res2: ', self.residual2(output1).size())
        # print('res1: ', self.residual1(inputs).size())


        
        outputs = self.up2(outputs + self.residual4(output3))
        # print('outputs: ', outputs.size())
        outputs = self.up3(outputs + self.residual3(output2))
        # print('outputs: ', outputs.size())
        outputs = self.up4(outputs + self.residual2(output1))
        # print('outputs: ', outputs.size())
        outputs = self.up5(outputs + self.residual1(inputs))
        # print('outputs: ', outputs.size())
        
        out_line = self.out_line(outputs)

        return out_line, pts, outputs


####################################################################
##
## lane_detection_network
##
####################################################################
class HGNet(nn.Module):
    def __init__(self):
        super(HGNet, self).__init__()

        # self.resizing = resize_layer(3, 128)

        #feature extraction
        self.layer1 = hourglass_block(3, 128)
        # self.layer2 = hourglass_block(128, 128)
        # self.conv = bottleneck(128, 512)
        # self.resize_up = resize_up(512, 1)


    def forward(self, inputs):
        #feature extraction
        # out = self.resizing(inputs)
        # print(out.size())
        out_line, pts, outputs = self.layer1(inputs)
        # result2, pts2, out = self.layer2(out)
        # inter = self.conv(out)
        # output = self.resize_up(inter)         

        return out_line, pts

# test = torch.rand((1, 3, 256, 256))
# print(test.size())

# model = HGNet()
# model.cuda()
# pts, output = model(test.cuda())
# print(pts.size())
# print(output.size())

