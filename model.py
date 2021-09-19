
import os
from os import walk
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from pytorch_lightning import LightningModule
from sklearn.metrics import roc_auc_score

from utils import blend, smooth_bce, utility_score


class PatternFinder(LightningModule):

    def __init__(self, in_channels, multf=2):

        super(PatternFinder, self).__init__()

        #TODO: anadir uno o varios canales con ruido

        #TODO: probar dos cosas:
        # - dos convoluciones: ultimo tramo y todo pero con el tiempo corriendo cada vez mas lento
        # - primera derivada

        #TODO: hay que visualizar para ver que podemos hacer

        #TODO: controlr kernel sizes con parametros a entrenar?

        # self.avgpool_b = nn.AvgPool1d(kernel_size=15, stride=3)
        
        self.block0 = Dablock(in_channels, multf)
        #self.block1 = Dablock(in_channels*3, multf//2)
        self.block1 = Dablock(in_channels, multf)
        self.block2 = Dablock(in_channels, multf, 2)

        input_width = in_channels*multf**5 * 3

        # self.batch_norm1 = nn.BatchNorm1d(input_width)
        # self.head = nn.Linear(input_width, input_width)

        self.linear = nn.Linear(input_width, 1)

    def forward(self, series):

        x0 = self.block0(series)

        #x1_a = F.avg_pool1d(series, kernel_size=3, stride=1, padding=1)
        x1_b = F.avg_pool1d(series, kernel_size=5, stride=1, padding=2)
        #x1_c = F.avg_pool1d(series, kernel_size=9, stride=1, padding=4)
        #x1 = self.block1(torch.hstack((x1_a, x1_b, x1_c)))
        x1 = self.block1(x1_b)

        x2 = self.block2(series)

        x = torch.hstack((x0,x1,x2))

        return self.linear(x), x

    def training_step(self, train_batch, batch_idx):

        series, stats, targets = train_batch

        logits = self.forward(series)[0]

        loss = torch.sqrt(torch.mean(torch.square((targets - logits.squeeze()) / targets), dim=0))

        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):

        series, stats, targets = val_batch

        logits = self.forward(series)[0]  #,stats.reshape(-1,1)

        loss = torch.sqrt(torch.mean(torch.square((targets - logits.squeeze()) / targets), dim=0))

        self.log('val_loss', loss)
        self.log('val_monit', loss.cpu().item())


    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
        sccheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=1/3, verbose=True, min_lr=1e-5)
        return [optimizer], {'scheduler': sccheduler, 'monitor': 'val_monit'}




class Dablock(LightningModule):

    def __init__(self, in_channels, multf, dilation=1):

        super(Dablock, self).__init__()

        self.batchnorm = nn.BatchNorm1d(in_channels)

        self.ks = 3 if dilation==1 else 2

        self.conv1 = nn.Conv1d(in_channels, in_channels*multf, kernel_size=self.ks, stride=1, padding=0, bias=True, dilation=dilation)
        self.conv2 = nn.Conv1d(in_channels*multf, in_channels*multf**2, kernel_size=self.ks, stride=1, padding=0, bias=True, dilation=dilation**2)
        self.conv3 = nn.Conv1d(in_channels*multf**2, in_channels*multf**3, kernel_size=self.ks, stride=1, padding=0, bias=True, dilation=dilation**3)

        self.conv4 = nn.Conv1d(in_channels*multf**3, in_channels*multf**4, kernel_size=self.ks, stride=1, padding=0, bias=True, dilation=dilation)
        self.conv5 = nn.Conv1d(in_channels*multf**3, in_channels*multf**4, kernel_size=3, stride=1, padding=0, bias=True, dilation=1)


    def forward(self, series):

        x = self.batchnorm(series)

        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = F.avg_pool1d(x, kernel_size=3, stride=3)

        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = F.avg_pool1d(x, kernel_size=3, stride=3)

        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = F.avg_pool1d(x, kernel_size=3, stride=3)

        if self.ks == 3:
            x = self.conv4(x)
            x = F.leaky_relu(x)
            x = F.avg_pool1d(x, kernel_size=3, stride=3)
        else:
            x = self.conv5(x)
            x = F.leaky_relu(x)
            x = F.avg_pool1d(x, kernel_size=2, stride=2)

        return torch.flatten(x, start_dim=1, end_dim=2)





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


