import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torchsummary import summary
from models.layers import vggUpconv

# vgg16 = models.vgg16_bn(pretrained=True)
# print(vgg16.features)
# feature_extractor = vgg16.features
# print(feature_extractor[0:5])
class driveNet(nn.Module):
    """Some Information about driveNet"""
    def __init__(self, in_channels=3, n_classes=2):
        super(driveNet, self).__init__()
        vgg16 = models.vgg16_bn(pretrained=True)
        feature_extractor = vgg16.features
        avgpool = vgg16.avgpool
        for param in feature_extractor.parameters():
            param.requires_grad = False
        for param in avgpool.parameters():
            param.requires_grad = True
        self.block1 = feature_extractor[0:7]
        self.block2 = feature_extractor[7:14]
        self.block3 = feature_extractor[14:24]
        self.block4 = feature_extractor[24:34]
        for param in self.block4.parameters():
            param.requires_grad = True
        self.block5 = feature_extractor[34:]
        for param in self.block5.parameters():
            param.requires_grad = True
        self.bottleneck = nn.Conv2d(512, 1024, 1, 1, 0)
        self.upsample1 = vggUpconv(1024,512, False)
        self.upsample2 = vggUpconv(512, 512)
        self.upsample3 = vggUpconv(512,256)
        self.upsample4 = vggUpconv(256, 128)
        self.upsample5 = vggUpconv(128, 64)
        self.upsample6 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.outconv1 = nn.Conv2d(64, 32, 3, 1, 1)
        # self.upsample1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        # self.upsample2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        # self.upsample3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        # self.upsample4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        # self.upsample5 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.outconv2 = nn.Conv2d(32, n_classes, 1, 1, 0)


    def forward(self, x):
        down1 = self.block1(x)
        down2 = self.block2(down1)
        down3 = self.block3(down2)
        down4 = self.block4(down3)
        down5 = self.block5(down4)
        bn = self.bottleneck(down5)
        up1 = self.upsample1(bn, down5)
        up2 = self.upsample2(up1, down4)
        up3 = self.upsample3(up2, down3)
        up4 = self.upsample4(up3, down2)
        up5 = self.upsample5(up4, down1)
        up6 = self.upsample6(up5)
        out1 = self.outconv1(up6)
        out2 = self.outconv2(out1)

        # up2 = self.upsample1
        # up
        # up1 = self.upsample1(bn)
        # up2 = self.upsample2(up1)
        # up3 = self.upsample3(up2)
        # up4 = self.upsample4(up3) 
        # up5 = self.upsample5(up4)       
        # out = self.outconv(up5)
        return out2

model = driveNet()
model.cuda()

summary(model, (3,224,224))