
import os
from os import walk
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from pytorch_lightning import LightningModule
from sklearn.metrics import roc_auc_score

class PatternFinder2(LightningModule):

    def __init__(self, in_channels, multf=2):

        super(PatternFinder2, self).__init__()

        self.batch_norm = nn.BatchNorm1d(in_channels)

        self.block00 = Dablock_ks3_notpooling(8, multf)
        self.block01 = Dablock_ks3_notpooling(10, multf)
        self.block02 = Dablock_ks3_notpooling(10, multf)

        self.block10 = Dablock_dilation_nopooling(8, multf)
        self.block11 = Dablock_dilation_nopooling(10, multf)
        self.block12 = Dablock_dilation_nopooling(10, multf)

        self.block_m0 = Dablock_merge(14, multf)
        self.block_m1 = Dablock_merge(14, multf)

        input_width = 56 * 4 * 2

        self.linear = nn.Linear(input_width, 1)

    def forward(self, series, stats):

        series = self.batch_norm(series)

        x00 = self.block00(series[:,:8])
        x01 = self.block01(series[:,8:18])
        x02 = self.block02(series[:,-10:])

        x0 = torch.cat((x00,x01,x02),dim=1)
        x0 = self.block_m0(x0)

        x10 = self.block10(series[:,:8])
        x11 = self.block11(series[:,8:18])
        x12 = self.block12(series[:,-10:])

        x1 = torch.cat((x10,x11,x12),dim=1)
        x1 = self.block_m1(x1)

        x = torch.hstack((x0,x1))

        return self.linear(x), x

    def training_step(self, train_batch, batch_idx):

        series, _, stats, targets = train_batch
        fut_rea_vol = targets[:,-4]

        logits = self.forward(series, stats)[0]

        #loss = torch.sqrt(torch.mean(torch.square((targets[:,-4] - logits.squeeze()) / targets[:,-4]), dim=0))
        loss = self.RMSE(logits.squeeze(), fut_rea_vol, 1 / (fut_rea_vol)**2)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):

        series, _, stats, targets = val_batch
        fut_rea_vol = targets[:,-4]

        logits = self.forward(series, stats)[0]  #,stats.reshape(-1,1)

        #loss = torch.sqrt(torch.mean(torch.square((targets[:,-4] - logits.squeeze()) / targets[:,-4]), dim=0))
        loss = self.RMSE(logits.squeeze(), fut_rea_vol, 1 / (fut_rea_vol)**2)

        val_rmspe = torch.sqrt(torch.mean(torch.square((fut_rea_vol - logits.squeeze()) / fut_rea_vol), dim=0)).cpu().item()

        self.log('val_loss', loss.cpu().item())
        self.log('val_rmspe', val_rmspe)


    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
        sccheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=1/3, verbose=True, min_lr=1e-5)
        return [optimizer], {'scheduler': sccheduler, 'monitor': 'val_rmspe'}

class Dablock_ks3_notpooling(LightningModule):

    def __init__(self, in_channels, multf):

        super(Dablock_ks3_notpooling, self).__init__()

        self.ks = 3
        self.st = 1

        self.conv1 = nn.Conv1d(in_channels, in_channels*multf, kernel_size=self.ks, stride=self.st)
        self.conv2 = nn.Conv1d(in_channels*multf, in_channels*multf**2, kernel_size=self.ks, stride=self.st)
        self.conv3 = nn.Conv1d(in_channels*multf**2, in_channels*multf, kernel_size=self.ks, stride=self.st)
        self.conv4 = nn.Conv1d(in_channels*multf, in_channels, kernel_size=self.ks, stride=self.st)
        self.conv5 = nn.Conv1d(in_channels, in_channels//2, kernel_size=self.ks, stride=self.st)

    def forward(self, series):

        x = series[:,:,-296:]

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

        return x

class Dablock_dilation_nopooling(LightningModule):

    def __init__(self, in_channels, multf, dilation=2):

        super(Dablock_dilation_nopooling, self).__init__()

        self.dilation = dilation
        self.ks = 3
        self.st = 1 #probar con distintos strides

        self.conv1 = nn.Conv1d(in_channels, in_channels*multf, kernel_size=self.ks, stride=self.st, dilation=dilation)
        self.conv2 = nn.Conv1d(in_channels*multf, in_channels*multf**2, kernel_size=self.ks, stride=self.st, dilation=dilation)
        self.conv3 = nn.Conv1d(in_channels*multf**2, in_channels*multf, kernel_size=self.ks, stride=self.st, dilation=dilation)

        if self.dilation == 2:
            self.conv4 = nn.Conv1d(in_channels*multf, in_channels//2, kernel_size=self.ks, stride=self.st, dilation=dilation)

    def forward(self, series):

        x = series[:,:,-295:]

        x = self.conv1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x)
        x = F.leaky_relu(x)

        if self.dilation == 2:
            x = self.conv4(x)
            x = F.leaky_relu(x)

        return x

class Dablock_merge(LightningModule):

    def __init__(self, in_channels, multf):

        super(Dablock_merge, self).__init__()

        self.ks = 3
        self.st = 1

        self.conv1 = nn.Conv1d(in_channels, in_channels*multf, kernel_size=self.ks, stride=self.st)
        self.conv2 = nn.Conv1d(in_channels*multf, in_channels*multf**2, kernel_size=self.ks, stride=self.st)
        self.conv3 = nn.Conv1d(in_channels*multf**2, in_channels*multf**2, kernel_size=self.ks, stride=self.st)

    def forward(self, series):

        x = series[:,:,-278:]

        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = F.avg_pool1d(x, kernel_size=3, stride=3)

        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = F.avg_pool1d(x, kernel_size=3, stride=3)

        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = F.avg_pool1d(x, kernel_size=7, stride=7)

        return torch.flatten(x, start_dim=1, end_dim=2)

class Dablock_dilation(LightningModule):

    #solo y con la mitad de los datos
    # en 3-e4 239
    # en 1-e5
    #train_loss=0.

    def __init__(self, in_channels, multf, dilation=2):

        super(Dablock_dilation, self).__init__()

        #self.batchnorm = nn.BatchNorm1d(in_channels)

        self.dilation = dilation
        self.ks = 3
        self.st = 1 #probar con distintos strides

        self.conv1 = nn.Conv1d(in_channels, in_channels*multf, kernel_size=self.ks, stride=self.st, dilation=dilation)
        self.conv2 = nn.Conv1d(in_channels*multf, in_channels*multf**2, kernel_size=self.ks, stride=self.st, dilation=dilation)
        self.conv3 = nn.Conv1d(in_channels*multf**2, in_channels*multf**3, kernel_size=self.ks, stride=self.st, dilation=dilation)

        if self.dilation == 2:
            self.conv4 = nn.Conv1d(in_channels*multf**3, in_channels*multf**3, kernel_size=self.ks, stride=self.st, dilation=dilation)

    def forward(self, series):

        #x = self.batchnorm(series[:,:,-295:])
        x = series[:,:,-295:]

        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = F.avg_pool1d(x, kernel_size=3, stride=3)

        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = F.avg_pool1d(x, kernel_size=3, stride=3)

        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = F.avg_pool1d(x, kernel_size=3, stride=3)

        if self.dilation == 2:
            x = self.conv4(x)
            x = F.leaky_relu(x)

        return torch.flatten(x, start_dim=1, end_dim=2)


