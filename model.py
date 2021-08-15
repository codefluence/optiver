
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

    def __init__(self, in_channels = 4):

        super(PatternFinder, self).__init__()

        self.batchnorm = nn.BatchNorm1d(in_channels)

        self.avgpool_a = nn.AvgPool1d(kernel_size=30, stride=6)
        self.avgpool_b = nn.AvgPool1d(kernel_size=15, stride=3)
        self.avgpool_c = nn.AvgPool1d(kernel_size=5, stride=1)

        multf = 4

        self.conv1_a = nn.Conv1d(in_channels, in_channels*multf, kernel_size=5, padding=1, bias=True)
        self.conv2_a = nn.Conv1d(in_channels*multf, in_channels*multf*2, kernel_size=5, padding=1, bias=True)
        self.conv3_a = nn.Conv1d(in_channels*multf*2, in_channels*multf, kernel_size=5, padding=1, bias=True)

        self.conv1_b = nn.Conv1d(in_channels, in_channels*multf, kernel_size=5, padding=1, bias=True)
        self.conv2_b = nn.Conv1d(in_channels*multf, in_channels*multf*2, kernel_size=5, padding=1, bias=True)
        self.conv3_b = nn.Conv1d(in_channels*multf*2, in_channels*multf, kernel_size=5, padding=1, bias=True)

        self.conv1_c = nn.Conv1d(in_channels, in_channels*multf, kernel_size=5, padding=1, bias=True)
        self.conv2_c = nn.Conv1d(in_channels*multf, in_channels*multf*2, kernel_size=5, padding=1, bias=True)
        self.conv3_c = nn.Conv1d(in_channels*multf*2, in_channels*multf, kernel_size=5, padding=1, bias=True)

        self.linear = nn.Linear(3*in_channels*multf*90, 1)  #3*4*96

        self.loss = nn.MSELoss()

    def forward(self, series, stats):

        x = self.batchnorm(series)
    
        x_a = self.avgpool_a(x)
        x_b = self.avgpool_b(x[:,:,-300:])
        x_c = self.avgpool_c(x[:,:,-100:])

        x_a = self.conv1_a(x_a)
        x_a = F.leaky_relu(x_a)
        x_a = self.conv2_a(x_a)
        x_a = F.leaky_relu(x_a)
        x_a = self.conv3_a(x_a)
        x_a = F.leaky_relu(x_a)

        x_b = self.conv1_b(x_b)
        x_b = F.leaky_relu(x_b)
        x_b = self.conv2_b(x_b)
        x_b = F.leaky_relu(x_b)
        x_b = self.conv3_b(x_b)
        x_b = F.leaky_relu(x_b)

        x_c = self.conv1_c(x_c)
        x_c = F.leaky_relu(x_c)
        x_c = self.conv2_c(x_c)
        x_c = F.leaky_relu(x_c)
        x_c = self.conv3_c(x_c)
        x_c = F.leaky_relu(x_c)

        x_a = torch.flatten(x_a, start_dim=1, end_dim=2)
        x_b = torch.flatten(x_b, start_dim=1, end_dim=2)
        x_c = torch.flatten(x_c, start_dim=1, end_dim=2)

        x = torch.hstack((x_a, x_b, x_c))

        return self.linear(x)

    def training_step(self, train_batch, batch_idx):

        series, stats, volatility = train_batch

        logits = self.forward(series, stats.reshape(-1,1))

        loss = self.loss(logits.squeeze(), volatility)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):

        series, stats, volatility = val_batch

        logits = self.forward(series, stats.reshape(-1,1))

        loss = self.loss(logits.squeeze(), volatility)

        v = volatility.cpu().numpy()
        p = logits.squeeze().cpu().numpy()
        rmspe = np.sqrt(np.mean(np.square((v - p) / v), axis=0))

        self.log('val_loss', loss)
        self.log('val_rmspe', rmspe)


    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
        sccheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=1/5, verbose=True, min_lr=1e-5)
        return [optimizer], {'scheduler': sccheduler, 'monitor': 'val_rmspe'}



class VolatilityClassifier(LightningModule):

    def __init__(self, input_width=600):

        super(VolatilityClassifier, self).__init__()

        in_channels = 10
        hidden_size = input_width

        self.batchnorm = nn.BatchNorm1d(in_channels)

        self.conv1 = nn.Conv1d(in_channels, in_channels*8, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv1d(in_channels*8, in_channels*8, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv1d(in_channels*8, in_channels*4, kernel_size=3, padding=1, bias=True)

        self.dense = nn.Linear(in_channels*4*600, 1)

        self.linear = nn.Linear(2, 1)

        self.loss = nn.MSELoss()

    def forward(self, series, stats):

        x = self.batchnorm(series)

        x = self.conv1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x)
        x = F.leaky_relu(x)

        x = torch.flatten(x, start_dim=1, end_dim=2)
        x = self.dense(x)

        x = torch.hstack((x,stats))

        return self.linear(x)

    def training_step(self, train_batch, batch_idx):

        series, stats, volatility = train_batch

        logits = self.forward(series, stats.reshape(-1,1))

        loss = self.loss(logits.squeeze(), volatility)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):

        series, stats, volatility = val_batch

        logits = self.forward(series, stats.reshape(-1,1))

        loss = self.loss(logits.squeeze(), volatility)

        v = volatility.cpu().numpy()
        p = logits.squeeze().cpu().numpy()
        rmspe = np.sqrt(np.mean(np.square((v - p) / v), axis=0))

        self.log('val_loss', loss)
        self.log('val_rmspe', rmspe)


    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
        sccheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=1/5, verbose=True, min_lr=1e-5)
        return [optimizer], {'scheduler': sccheduler, 'monitor': 'val_rmspe'}
