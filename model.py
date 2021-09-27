
import os
from os import walk
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from pytorch_lightning import LightningModule
from sklearn.metrics import roc_auc_score


class PatternFinder(LightningModule):

    def __init__(self, in_channels, medians, multf=1):

        super(PatternFinder, self).__init__()

        self.medians = torch.tensor(medians, device='cuda')

        self.batch_norm = nn.BatchNorm1d(in_channels)

        self.block0 = Dablock_ks3(in_channels, multf)
        self.block1 = Dablock_eye_ks3(in_channels, multf)
        #self.block2 = Dablock_dilation(in_channels, multf)

        input_width = in_channels*multf**3 * (4 + 5) #+ 3

        #self.head = nn.Linear(input_width, input_width)
        self.linear = nn.Linear(input_width, 1)

    def forward(self, series, stats):

        series = self.batch_norm(series)

        x0 = self.block0(series)   # - self.medians
        x1 = self.block1(series)
        #x2 = self.block2(series)

        #x0_c = self.block0_c(series)

        ###########x1_a = F.avg_pool1d(series, kernel_size=3, stride=1, padding=1)
        #x1_b = F.avg_pool1d(series, kernel_size=5, stride=1, padding=2)
        #x1 = self.block1(x1_b)
        ###########x1_c = F.avg_pool1d(series, kernel_size=9, stride=1, padding=4)
        ###########x1 = self.block1(torch.hstack((x1_a, x1_b, x1_c)))

        #x2 = self.block2(series)

        x = torch.hstack((x0,x1))

        # #x = F.dropout(x, p=0.1)
        # x = self.head(x)
        # x = F.leaky_relu(x)

        return self.linear(x), x

    def training_step(self, train_batch, batch_idx):

        series, _, stats, targets = train_batch

        logits = self.forward(series, stats)[0]

        loss = torch.sqrt(torch.mean(torch.square((targets[:,-4] - logits.squeeze()) / targets[:,-4]), dim=0))

        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):

        series, _, stats, targets = val_batch

        logits = self.forward(series, stats)[0]  #,stats.reshape(-1,1)

        loss = torch.sqrt(torch.mean(torch.square((targets[:,-4] - logits.squeeze()) / targets[:,-4]), dim=0))

        self.log('val_loss', loss)
        self.log('val_monit', loss.cpu().item())


    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
        sccheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=1/3, verbose=True, min_lr=1e-5)
        return [optimizer], {'scheduler': sccheduler, 'monitor': 'val_monit'}



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

class Dablock_ks3(LightningModule):

    #solo y con la mitad de los datos
    #240 en 1-e4
    #239 en 1-e5
    #train_loss=0.233

    #mejor con 4 u 8 salidas: alcanza 0.238

    def __init__(self, in_channels, multf):

        super(Dablock_ks3, self).__init__()

        #self.batchnorm = nn.BatchNorm1d(in_channels)

        self.ks = 3
        self.st = 1

        self.conv1 = nn.Conv1d(in_channels, in_channels*multf, kernel_size=self.ks, stride=self.st)
        self.conv2 = nn.Conv1d(in_channels*multf, in_channels*multf**2, kernel_size=self.ks, stride=self.st)
        self.conv3 = nn.Conv1d(in_channels*multf**2, in_channels*multf**3, kernel_size=self.ks, stride=self.st)
        self.conv4 = nn.Conv1d(in_channels*multf**3, in_channels*multf**3, kernel_size=self.ks, stride=self.st)

    def forward(self, series):

        #x = F.max_pool1d(series, kernel_size=5, stride=1)

        #x = self.batchnorm(x[:,:,-296:])
        x = series[:,:,-296:]

        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = F.avg_pool1d(x, kernel_size=3, stride=3)

        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = F.avg_pool1d(x, kernel_size=3, stride=3)

        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = F.avg_pool1d(x, kernel_size=3, stride=3)

        x = self.conv4(x)
        x = F.leaky_relu(x)
        x = F.avg_pool1d(x, kernel_size=2, stride=2)

        return torch.flatten(x, start_dim=1, end_dim=2)

class Dablock_eye_ks3(LightningModule):

#     #solo y con la mitad de los datos
#     #241 en 3.3-e4
#     #train_loss=0.235

    def __init__(self, in_channels, multf):

        super(Dablock_eye_ks3, self).__init__()

        #self.batchnorm = nn.BatchNorm1d(in_channels)

        self.ks = 3 #TODO:probar con 2
        self.st = 1

        self.conv1 = nn.Conv1d(in_channels, in_channels*multf, kernel_size=self.ks, stride=self.st, dilation=2)
        self.conv2 = nn.Conv1d(in_channels*multf, in_channels*multf**2, kernel_size=self.ks, stride=self.st, dilation=2**2)
        self.conv3 = nn.Conv1d(in_channels*multf**2, in_channels*multf**3, kernel_size=self.ks, stride=self.st, dilation=2**3)

    def forward(self, series):

        #x = self.batchnorm(series[:,:,-262:])
        x = series[:,:,-262:]

        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = F.avg_pool1d(x, kernel_size=3, stride=3)

        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = F.avg_pool1d(x, kernel_size=3, stride=3)

        x = self.conv3(x)
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








############################################################################################################

############################################################################################################

############################################################################################################


class VolatilityClassifier(LightningModule):

    def __init__(self, input_width):

        super(VolatilityClassifier, self).__init__()

        #self.dense0 = nn.Linear(4320, 8)

        hidden_size = input_width*2

        self.batch_norm1 = nn.BatchNorm1d(input_width)
        self.dense1 = nn.Linear(input_width, hidden_size)

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size)

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dense3 = nn.Linear(hidden_size, hidden_size)

        self.linear = nn.Linear(hidden_size, 1)

        self.loss = nn.MSELoss()

    def forward(self, x):

        x = self.batch_norm1(x)
        x = self.dense1(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=0.2)

        x = self.batch_norm2(x)
        x = self.dense2(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=0.2)

        x = self.batch_norm3(x)
        x = self.dense3(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=0.2)

        return self.linear(x)

    def training_step(self, train_batch, batch_idx):

        _, stats, targets = train_batch

        preds = self.forward(stats)

        loss = torch.sqrt(torch.mean(torch.square((targets - preds.squeeze()) / targets), dim=0))

        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):

        _, stats, targets = val_batch

        preds = self.forward(stats)

        loss = torch.sqrt(torch.mean(torch.square((targets - preds.squeeze()) / targets), dim=0))

        self.log('val_loss', loss)
        self.log('val_monit', loss.cpu().item())


    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
        sccheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=1/3, verbose=True, min_lr=1e-5)
        return [optimizer], {'scheduler': sccheduler, 'monitor': 'val_monit'}


