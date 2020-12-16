""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)


    def forward(self, x,embedding=None, encode_opt=True):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x_emb =  self.down4(x4)
        x5 = x_emb if encode_opt else embedding
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if encode_opt:
            return logits, x_emb
        return logits,embedding
    
    # def forward_old(self, x,embedding=None):
    #     x1 = self.inc(x)
    #     x2 = self.down1(x1)
    #     x3 = self.down2(x2)
    #     x4 = self.down3(x3)
    #     x_emb =  self.down4(x4)
    #     print(self.encode_opt)
    #     if self.encode_opt:
    #         x5 = x_emb
    #         print('x_emb')
    #     else:
    #         x5= embedding
    #         x4= torch.zeros_like(x4)
    #         x3= torch.zeros_like(x3)
    #         x2= torch.zeros_like(x2)
    #         x1= torch.zeros_like(x1)
    #         print('embedding')

    #     x = self.up1(x5, x4)
    #     x = self.up2(x, x3)
    #     x = self.up3(x, x2)
    #     x = self.up4(x, x1)
    #     logits = self.outc(x)
    #     if self.encode_opt:
    #         return logits, x_emb
    #     return logits,embedding
