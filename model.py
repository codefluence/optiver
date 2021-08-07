
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


    # @staticmethod
    # def predict(X, split):

    #     path = './weights/trend_'+split+'.ckpt'

    #     if os.path.isfile(path):

    #         tre_model = TrendClassifier.load_from_checkpoint(path, input_width=X.shape[1])
    #         tre_model.cpu()
    #         tre_model.eval()

    #         tre_test = torch.tensor(X, dtype=torch.float32, requires_grad=False, device='cpu')
    #         return torch.sigmoid(tre_model(tre_test)).detach().numpy().squeeze()

    #     return None


    # @staticmethod
    # def predict_blend(X):

    #     trend_pred = None

    #     if os.path.isdir('./weights/trend/'):

    #         tre_files = walk('./weights/trend/')

    #         root, _, tre_model_paths = next(tre_files)
    #         tre_test = torch.tensor(X, dtype=torch.float32, requires_grad=False, device='cpu')
    #         tre_num_models = 0

    #         for model_path in tre_model_paths:

    #             tre_num_models += 1

    #             if tre_num_models == 1:
    #                 trend_pred = np.zeros((tre_test.shape[0])).astype(np.float32) 

    #             tre_model = TrendClassifier.load_from_checkpoint(root+model_path, input_width=X.shape[1])
    #             tre_model.cpu()
    #             tre_model.eval()

    #             cvpred = torch.sigmoid(tre_model(tre_test)).detach().numpy().squeeze()
    #             trend_pred += cvpred

    #         trend_pred /= tre_num_models

    #     return trend_pred

