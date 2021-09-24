
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

    def __init__(self, in_channels, stats_num, series_medians, multf=2):

        super(PatternFinder, self).__init__()

        self.series_medians = torch.tensor(series_medians, device='cuda')

        #self.block1 = Dablock(in_channels*3, multf//2)

        self.block0 = Dablock(in_channels, multf)
        #self.block1 = Dablock(in_channels, multf)
        #self.block2 = Dablock(in_channels, multf, dilation=2)
        #self.block3 = Dablock(in_channels, multf)

        input_width = sum(in_channels * multf * [4*3]) + stats_num  #,3*2

        di = multf*2
        self.head = nn.Linear(input_width, input_width//di)
        self.linear = nn.Linear(input_width//di, 1)

        #self.loss = nn.MSELoss()

    def RMSE(self, input, target, weight):

        return torch.sqrt(torch.sum(weight * (input - target)**2) / torch.sum(weight))

    def forward(self, series, stats):

        x0 = self.block0(series)
        #x1 = self.block1(series - self.series_medians) #TODO

        #x1_a = F.avg_pool1d(series, kernel_size=3, stride=1, padding=1)
        #x1_b = F.avg_pool1d(series, kernel_size=5, stride=1, padding=2)
        #x1_c = F.avg_pool1d(series, kernel_size=9, stride=1, padding=4)
        #x1 = self.block1(torch.hstack((x1_a, x1_b, x1_c)))
        #x1 = self.block1(x1_b)

        #x2 = self.block2(series)

        # x3 = F.max_pool1d(series, kernel_size=10, stride=1)
        # x3 = F.avg_pool1d(x3, kernel_size=7, stride=1)
        # x3 = self.block3(x3)

        x = torch.hstack((x0,stats))

        x = self.head(x)
        x = F.leaky_relu(x)
        #x = F.dropout(x, p=0.2)

        return self.linear(x), x

    def training_step(self, train_batch, batch_idx):

        series, stats, targets = train_batch

        fut_rea_vol  = targets[:,-4]
        logits = self.forward(series, stats)[0]

        loss = torch.sqrt(torch.mean(torch.square((fut_rea_vol - logits.squeeze()) / fut_rea_vol), dim=0))
        #loss = self.loss(logits.squeeze(), delta)
        #loss = self.RMSE(logits.squeeze(), fut_rea_vol, 1 / (fut_rea_vol)**2)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):

        series, stats, targets = val_batch

        increase     = targets[:,-1]
        delta        = targets[:,-2]
        past_rea_vol = targets[:,-3]
        fut_rea_vol  = targets[:,-4]

        #TODO: nograd needed?

        #fut_rea_vol_hat
        logits = self.forward(series, stats)[0]  #,stats.reshape(-1,1)

        loss = torch.sqrt(torch.mean(torch.square((fut_rea_vol - logits.squeeze()) / fut_rea_vol), dim=0)).cpu().item()
        #loss = self.loss(logits.squeeze(), delta).cpu().item()
        #loss = self.RMSE(logits.squeeze(), fut_rea_vol, 1 / (fut_rea_vol)**2).cpu().item()

        #fut_rea_vol_estim = past_rea_vol + logits.squeeze()
        fut_rea_vol_estim = logits.squeeze()
        val_increase_mae_estim = fut_rea_vol_estim / past_rea_vol - 1
        val_increase_mae = torch.mean(torch.abs(val_increase_mae_estim - increase)).cpu().item()

        val_rmspe = torch.sqrt(torch.mean(torch.square((fut_rea_vol - fut_rea_vol_estim) / fut_rea_vol), dim=0)).cpu().item()

        self.log('val_loss', loss)
        self.log('val_increase_mae', val_increase_mae)
        self.log('val_rmspe', val_rmspe)

        if batch_idx==0:
            print()
            print(  f'\033[93m-------',
                    'val_rmspe', np.round(val_rmspe,3),
                    '| val_loss', np.round(loss,7),
                    '| val_increase_mae',np.round(val_increase_mae,2),
                    '| mean(real_increase)',np.round(torch.mean(torch.abs(increase)).cpu().item(),2),
                    'median(real_increase)',np.round(torch.median(torch.abs(increase)).cpu().item(),2),
                    f'-------\033[0m')


    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        sccheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=1/5, verbose=True, min_lr=1e-6)
        return [optimizer], {'scheduler': sccheduler, 'monitor': 'val_rmspe'}




class Dablock(LightningModule):

    def __init__(self, in_channels, multf, kernel_size=3, dilation=1, ch_dropout=0):

        super(Dablock, self).__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.ch_dropout = ch_dropout

        self.batchnorm0 = nn.BatchNorm1d(in_channels)
        # self.batchnorm1 = nn.BatchNorm1d(in_channels*multf)
        # self.batchnorm2 = nn.BatchNorm1d(in_channels*multf*2)
        # self.batchnorm3 = nn.BatchNorm1d(in_channels*multf*3)

        self.conv00 = nn.Conv1d(in_channels, in_channels*multf, kernel_size=kernel_size, dilation=dilation)
        #self.conv01 = nn.Conv1d(in_channels*multf, in_channels*multf, kernel_size=kernel_size, dilation=dilation)

        self.conv10 = nn.Conv1d(in_channels*multf, in_channels*multf*2, kernel_size=kernel_size, dilation=dilation**2)
        #self.conv11 = nn.Conv1d(in_channels*multf*2, in_channels*multf*2, kernel_size=kernel_size, dilation=dilation**2)

        self.conv20 = nn.Conv1d(in_channels*multf*2, in_channels*multf*3, kernel_size=kernel_size, dilation=dilation**3)
        #self.conv21 = nn.Conv1d(in_channels*multf*3, in_channels*multf*3, kernel_size=kernel_size, dilation=dilation**3)

        self.conv30 = nn.Conv1d(in_channels*multf*3, in_channels*multf*4, kernel_size=kernel_size)
        #self.conv31 = nn.Conv1d(in_channels*multf*4, in_channels*multf*4, kernel_size=kernel_size)

        #self.conv3b = nn.Conv1d(in_channels*multf*3, in_channels*multf*4, kernel_size=3, stride=3, bias=True)

    def channel_dropout(self, x):

        if self.ch_dropout>0:

            return x * torch.tensor(np.random.choice(2, x.shape[:2], p=[self.ch_dropout, 1-self.ch_dropout]),
                                    dtype=torch.float32, device='cuda').unsqueeze(2)
        else:
            return x

    def forward(self, series):

        x = self.batchnorm0(series)

        x = self.conv00(x)
        x = F.leaky_relu(x)
        # x = self.conv01(x)
        # x = F.leaky_relu(x)
        x = F.avg_pool1d(x, kernel_size=2, stride=2)#TODO kernel y stride iguales?
        #x = self.channel_dropout(x)

        #x = self.batchnorm1(x)
        x = self.conv10(x)
        x = F.leaky_relu(x)
        # x = self.conv11(x)
        # x = F.leaky_relu(x)
        x = F.avg_pool1d(x, kernel_size=2, stride=2)
        #x = self.channel_dropout(x)

        #x = self.batchnorm2(x)
        x = self.conv20(x)
        x = F.leaky_relu(x)
        # x = self.conv21(x)
        # x = F.leaky_relu(x)
        x = F.avg_pool1d(x, kernel_size=2, stride=2)
        #x = self.channel_dropout(x)

        if self.dilation == 1:
            #x = self.batchnorm2(x)
            x = self.conv30(x)
            x = F.leaky_relu(x)
            # x = self.conv31(x)
            # x = F.leaky_relu(x)
            x = F.avg_pool1d(x, kernel_size=3, stride=3)
            #x = self.channel_dropout(x)

        #x = self.batchnorm3(x)
        # if self.dilation == 1:
        #     x = self.conv30(x)
        #     x = F.leaky_relu(x)
        #     # x = self.conv31(x)
        #     # x = F.leaky_relu(x)
        #     x = F.avg_pool1d(x, kernel_size=self.ks+1, stride=self.ks+1)
        # else:
        #     x = self.conv3b(x)
        #     x = F.leaky_relu(x)
        #     x = F.avg_pool1d(x, kernel_size=4, stride=4)
        #x = self.channel_dropout(x)

        return torch.flatten(x, start_dim=1, end_dim=2)










###################################################################################################################
###################################################################################################################

###################################################################################################################

###################################################################################################################
###################################################################################################################







class VolatilityClassifier(LightningModule):

    def __init__(self, input_width):

        super(VolatilityClassifier, self).__init__()

        hidden_size = input_width*8

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
        #x = F.dropout(x, p=0.1)

        x = self.batch_norm2(x)
        x = self.dense2(x)
        x = F.leaky_relu(x)
        #x = F.dropout(x, p=0.1)

        x = self.batch_norm3(x)
        x = self.dense3(x)
        x = F.leaky_relu(x)
        #x = F.dropout(x, p=0.1)

        return self.linear(x)

    def training_step(self, train_batch, batch_idx):

        _, _, stats, targets = train_batch

        delta = targets[:,-2]
        logits = self.forward(stats)

        loss = self.loss(logits.squeeze(), delta)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):

        _, _, stats, targets = val_batch

        delta = targets[:,-2]
        logits = self.forward(stats)

        loss = self.loss(logits.squeeze(), delta).cpu().item()

        mae = torch.mean(torch.abs(logits.squeeze() - delta)).cpu().item()

        self.log('val_loss', loss)
        self.log('val_mae', mae)
        self.log('val_rmspe', mae)

        #TODO: nograd needed?
        past_rea_vol = targets[:,-3]
        fut_rea_vol  = targets[:,-4]
        fut_rea_vol_estim = past_rea_vol + logits.squeeze()
        rmspe = torch.sqrt(torch.mean(torch.square((fut_rea_vol - fut_rea_vol_estim) / fut_rea_vol), dim=0)).cpu().item()
        self.log('val_rmspe', rmspe)

        if batch_idx==0:
            print()
            print(  f'\033[93m-------',
                    'val_rmspe', np.round(rmspe,3),
                    '| val_loss', np.round(loss,7),
                    '| val_mae', np.round(mae,7),
                    '| mean(truth)',np.round(torch.mean(torch.abs(delta)).cpu().item(),7),
                    'median(truth)',np.round(torch.median(torch.abs(delta)).cpu().item(),7),
                    f'-------\033[0m')


    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
        sccheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=1/3, verbose=True, min_lr=1e-5)
        return [optimizer], {'scheduler': sccheduler, 'monitor': 'val_rmspe'}


