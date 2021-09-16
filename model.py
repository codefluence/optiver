
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


class LiquidityTrend(LightningModule):

    def __init__(self, in_channels=4, multf=4):

        super(LiquidityTrend, self).__init__()

        self.batchnorm = nn.BatchNorm1d(in_channels)
        
        self.conv1_a = nn.Conv1d(in_channels, in_channels*multf, kernel_size=4, stride=4, padding=0, bias=True)
        self.conv2_a = nn.Conv1d(in_channels*multf, in_channels*multf*2, kernel_size=3, stride=3, padding=0, bias=True)
        self.conv3_a = nn.Conv1d(in_channels*multf*2, in_channels*multf*2*2, kernel_size=3, stride=3, padding=0, bias=True)

        self.conv1_b = nn.Conv1d(in_channels, in_channels*multf, kernel_size=3, stride=3, padding=0, bias=True)
        self.conv2_b = nn.Conv1d(in_channels*multf, in_channels*multf*2, kernel_size=3, stride=3, padding=0, bias=True)
        self.conv3_b = nn.Conv1d(in_channels*multf*2, in_channels*multf*2*2, kernel_size=3, stride=3, padding=0, bias=True)

        self.conv1_c = nn.Conv1d(in_channels, in_channels*multf, kernel_size=2, stride=2, padding=0, bias=True)
        self.conv2_c = nn.Conv1d(in_channels*multf, in_channels*multf*2, kernel_size=2, stride=2, padding=0, bias=True)
        self.conv3_c = nn.Conv1d(in_channels*multf*2, in_channels*multf*2*2, kernel_size=2, stride=2, padding=0, bias=True)

        self.linear = nn.Linear(in_channels*multf*2*2 * 1 * 3,1)  #in_channels*multf*2*2

        self.loss = nn.MSELoss()

    def forward(self, series):

        x = self.batchnorm(series)
    
        x_a = x[:,:,-36:]
        x_b = x[:,:,-27:]
        x_c = x[:,:,-8:]

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

        return self.linear(x), x

    def training_step(self, train_batch, batch_idx):

        series, targets = train_batch

        logits = self.forward(series)[0]

        loss = self.loss(logits.squeeze(), targets)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):

        series, targets = val_batch

        logits = self.forward(series)[0]

        loss = self.loss(logits.squeeze(), targets)

        self.log('val_loss', loss)
        self.log('val_monit', loss.detach().cpu().item() * 1e5)


    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
        sccheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=1/3, verbose=True, min_lr=1e-5)
        return [optimizer], {'scheduler': sccheduler, 'monitor': 'val_monit'}





class PatternFinder(LightningModule):

    def __init__(self, in_channels=4, multf=4):

        super(PatternFinder, self).__init__()

        #TODO: anadir uno o varios canales con ruido

        #TODO: probar dos cosas:
        # - dos convoluciones: ultimo tramo y todo pero con el tiempo corriendo cada vez mas lento
        # - primera derivada

        #TODO: hay que visualizar para ver que podemos hacer

        self.batchnorm = nn.BatchNorm1d(in_channels)

        # self.avgpool_b = nn.AvgPool1d(kernel_size=15, stride=3)
        # self.avgpool_c = nn.AvgPool1d(kernel_size=5, stride=1)
        
        self.conv1_a = nn.Conv1d(in_channels, in_channels*multf, kernel_size=4, stride=4, padding=0, bias=True)
        self.conv2_a = nn.Conv1d(in_channels*multf, in_channels*multf*2, kernel_size=4, stride=4, padding=0, bias=True)
        self.conv3_a = nn.Conv1d(in_channels*multf*2, in_channels*multf*2*2, kernel_size=4, stride=4, padding=0, bias=True)

        self.conv1_b = nn.Conv1d(in_channels, in_channels*multf, kernel_size=3, stride=3, padding=0, bias=True)
        self.conv2_b = nn.Conv1d(in_channels*multf, in_channels*multf*2, kernel_size=3, stride=3, padding=0, bias=True)
        self.conv3_b = nn.Conv1d(in_channels*multf*2, in_channels*multf*2*2, kernel_size=3, stride=3, padding=0, bias=True)

        self.conv1_c = nn.Conv1d(in_channels, in_channels*multf, kernel_size=2, stride=2, padding=0, bias=True)
        self.conv2_c = nn.Conv1d(in_channels*multf, in_channels*multf*2, kernel_size=2, stride=2, padding=0, bias=True)
        self.conv3_c = nn.Conv1d(in_channels*multf*2, in_channels*multf*2*2, kernel_size=2, stride=2, padding=0, bias=True)

        self.linear = nn.Linear(in_channels*multf*2*2 * 1 * 3,1)  #in_channels*multf*2*2


    def forward(self, series):

        x = self.batchnorm(series)
    
        x_a = x[:,:,-64:]
        x_b = x[:,:,-27:]
        x_c = x[:,:,-8:]

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

        # x_a = x_a[:,:,1] - x_a[:,:,0]
        # x_c = x_c[:,:,1] - x_c[:,:,0]

        x_a = torch.flatten(x_a, start_dim=1, end_dim=2)
        x_b = torch.flatten(x_b, start_dim=1, end_dim=2)
        x_c = torch.flatten(x_c, start_dim=1, end_dim=2)

        x = torch.hstack((x_a, x_b, x_c))

        return self.linear(x), x

    def training_step(self, train_batch, batch_idx):

        series, _, targets = train_batch

        logits = self.forward(series)[0]  #, stats.reshape(-1,1)

        loss = torch.sqrt(torch.mean(torch.square((targets - logits.squeeze()) / targets), dim=0))

        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):

        series, _, targets = val_batch

        logits = self.forward(series)[0]  #,stats.reshape(-1,1)

        loss = torch.sqrt(torch.mean(torch.square((targets - logits.squeeze()) / targets), dim=0))

        self.log('val_loss', loss)
        self.log('val_monit', loss.cpu().item())


    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
        sccheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=1/3, verbose=True, min_lr=1e-5)
        return [optimizer], {'scheduler': sccheduler, 'monitor': 'val_monit'}



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


