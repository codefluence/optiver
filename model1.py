
import os
from os import walk
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from pytorch_lightning import LightningModule



class PatternFinder1(LightningModule):

    def __init__(self, in_channels, multf=2):

        super(PatternFinder1, self).__init__()

        self.batch_norm_1d = nn.BatchNorm1d(in_channels)
        # self.batch_norm_2d = nn.BatchNorm2d(in_maps)

        # self.conv2d_a = nn.Conv2d(in_maps, 32, kernel_size=3, stride=1)
        # self.conv2d_b = nn.Conv2d(32, 4, kernel_size=2, stride=1)

        self.block00 = Dablock_ks3(10, multf)
        self.block01 = Dablock_ks3(10, multf)
        self.block02 = Dablock_ks3(7, multf)

        self.block10 = Dablock_eye_ks3(10, multf)
        self.block11 = Dablock_eye_ks3(10, multf)
        self.block12 = Dablock_eye_ks3(7, multf)

        input_width = 1080

        self.head = nn.Linear(input_width, 32)
        self.linear = nn.Linear(32, 1)

    def RMSE(self, input, target, weight):

        return torch.sqrt(torch.sum(weight * (input - target)**2) / torch.sum(weight))

    def forward(self, series):

        # xmaps = self.conv2d_a(self.batch_norm_2d(maps))
        # xmaps = self.conv2d_b(xmaps).squeeze(2)
        # x = torch.cat(( xmaps, self.batch_norm_1d(series[:,:,-297:])),dim=1)

        x = self.batch_norm_1d(series)

        c0=10
        c1=10+c0
        c2=7+c1

        x00 = self.block00(x[:,:c0])
        x01 = self.block01(x[:,c0:c1])
        x02 = self.block02(x[:,c1:c2])

        x10 = self.block10(x[:,:c0])
        x11 = self.block11(x[:,c0:c1])
        x12 = self.block12(x[:,c1:c2])

        xall = torch.hstack((x00,x01,x02, x10,x11,x12))

        xall = self.head(xall)
        xall = F.leaky_relu(xall)
        x = F.dropout(x, p=0.1)

        return self.linear(xall)

    def training_step(self, train_batch, batch_idx):

        series, _, targets = train_batch
        fut_rea_vol = targets[:,-4]

        logits = self.forward(series)

        loss = torch.sqrt(torch.mean(torch.square((fut_rea_vol - logits.squeeze()) / fut_rea_vol), dim=0))
        #loss = self.RMSE(logits.squeeze(), fut_rea_vol, 1 / (fut_rea_vol)**2)

        self.log('train_loss', loss.cpu().item())

        return loss

    def validation_step(self, val_batch, batch_idx):

        series, _, targets = val_batch
        fut_rea_vol = targets[:,-4]

        logits = self.forward(series)

        loss = torch.sqrt(torch.mean(torch.square((fut_rea_vol - logits.squeeze()) / fut_rea_vol), dim=0))
        #loss = self.RMSE(logits.squeeze(), fut_rea_vol, 1 / (fut_rea_vol)**2)

        val_rmspe = torch.sqrt(torch.mean(torch.square((fut_rea_vol - logits.squeeze()) / fut_rea_vol), dim=0)).cpu().item()

        self.log('val_loss', loss.cpu().item())
        self.log('val_rmspe', val_rmspe)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
        sccheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=1/3, verbose=True, min_lr=1e-5)
        return [optimizer], {'scheduler': sccheduler, 'monitor': 'val_rmspe'}



class Dablock_ks3(LightningModule):

    def __init__(self, in_channels, multf):

        super(Dablock_ks3, self).__init__()

        self.ks = 3
        self.st = 1

        self.conv1 = nn.Conv1d(in_channels, in_channels*multf, kernel_size=self.ks, stride=self.st)
        self.conv2 = nn.Conv1d(in_channels*multf, in_channels*multf**2, kernel_size=self.ks, stride=self.st)
        self.conv3 = nn.Conv1d(in_channels*multf**2, in_channels*multf**2, kernel_size=self.ks, stride=self.st)
        self.conv4 = nn.Conv1d(in_channels*multf**2, in_channels*multf**2, kernel_size=self.ks, stride=self.st)

    def forward(self, series):

        x = series[:,:,:]#-296

        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = F.avg_pool1d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = F.avg_pool1d(x, kernel_size=2, stride=2)

        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = F.avg_pool1d(x, kernel_size=2, stride=2)

        x = self.conv4(x)
        x = F.leaky_relu(x)
        x = F.avg_pool1d(x, kernel_size=3, stride=3)

        return torch.flatten(x, start_dim=1, end_dim=2)


class Dablock_eye_ks3(LightningModule):

    def __init__(self, in_channels, multf, pooling=True):

        super(Dablock_eye_ks3, self).__init__()

        self.pooling = pooling

        self.ks = 3 #TODO:probar con 2
        self.st = 1

        self.conv1 = nn.Conv1d(in_channels, in_channels*multf, kernel_size=self.ks, stride=self.st, dilation=2)
        self.conv2 = nn.Conv1d(in_channels*multf, in_channels*multf**2, kernel_size=self.ks, stride=self.st, dilation=2**2)
        self.conv3 = nn.Conv1d(in_channels*multf**2, in_channels*multf**2, kernel_size=self.ks, stride=self.st, dilation=2**3)

    def forward(self, series):

        x = series[:,:,:]#-262

        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = F.avg_pool1d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = F.avg_pool1d(x, kernel_size=2, stride=2)

        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = F.avg_pool1d(x, kernel_size=3, stride=3)

        return torch.flatten(x, start_dim=1, end_dim=2)




class Dablock_long(LightningModule):

    #no es de los mejores
    #0.242 en epoch 13 a 3.3e-4

    def __init__(self, in_channels, multf, dilation=1):

        super(Dablock_long, self).__init__()

        self.batchnorm = nn.BatchNorm1d(in_channels)

        self.ks = 3
        self.st = 2

        self.conv1 = nn.Conv1d(in_channels, in_channels*multf, kernel_size=self.ks, stride=self.st)
        self.conv2 = nn.Conv1d(in_channels*multf, in_channels*multf**2, kernel_size=self.ks, stride=self.st)
        self.conv3 = nn.Conv1d(in_channels*multf**2, in_channels*multf**3, kernel_size=self.ks, stride=self.st)
        self.conv4 = nn.Conv1d(in_channels*multf**3, in_channels*multf**3, kernel_size=self.ks, stride=self.st)
        self.conv5 = nn.Conv1d(in_channels*multf**3, in_channels*multf**3, kernel_size=self.ks, stride=self.st)

    def forward(self, series):

        x = self.batchnorm(series[:,:,-296:])

        x = self.conv1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x)
        x = F.leaky_relu(x)

        x = self.conv4(x)
        x = F.leaky_relu(x)

        x = self.conv5(x)
        x = F.leaky_relu(x)

        x = F.avg_pool1d(x, kernel_size=2, stride=2)

        return torch.flatten(x, start_dim=1, end_dim=2)







# class Dablock_ks2(LightningModule):

#     #solo y con la mitad de los datos
#     #241 en 1-e4
#     #240 en 1-e5
#     #train_loss=0.235

#     def __init__(self, in_channels, multf, dilation=1):

#         super(Dablock_ks2, self).__init__()

#         self.batchnorm = nn.BatchNorm1d(in_channels)

#         self.ks = 2
#         self.st = 1

#         self.conv1 = nn.Conv1d(in_channels, in_channels*multf, kernel_size=self.ks, stride=self.st)
#         self.conv2 = nn.Conv1d(in_channels*multf, in_channels*multf**2, kernel_size=self.ks, stride=self.st)
#         self.conv3 = nn.Conv1d(in_channels*multf**2, in_channels*multf**3, kernel_size=self.ks, stride=self.st)
#         self.conv4 = nn.Conv1d(in_channels*multf**3, in_channels*multf**3, kernel_size=self.ks, stride=self.st)

#     def forward(self, series):

#         x = self.batchnorm(series[:,:,-256:])

#         x = self.conv1(x)
#         x = F.leaky_relu(x)
#         x = F.avg_pool1d(x, kernel_size=3, stride=3)

#         x = self.conv2(x)
#         x = F.leaky_relu(x)
#         x = F.avg_pool1d(x, kernel_size=3, stride=3)

#         x = self.conv3(x)
#         x = F.leaky_relu(x)
#         x = F.avg_pool1d(x, kernel_size=3, stride=3)

#         x = self.conv4(x)
#         x = F.leaky_relu(x)
#         x = F.avg_pool1d(x, kernel_size=4, stride=4)

#         return torch.flatten(x, start_dim=1, end_dim=2)


