from datetime import datetime

import torch
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from data import OptiverDataModule
from model import VolatilityClassifier, PatternFinder

def fit_model():

    data = OptiverDataModule()

    model = PatternFinder(data.series.shape[1])
    # model = VolatilityClassifier(data.stats.shape[1])

    filename = 'optiver-{epoch}-{val_rmspe:.4f}'
    dirpath='./weights/'

    print('time start:',datetime.now().strftime("%H:%M:%S"))

    early_stop_callback = EarlyStopping(
        monitor='val_rmspe',
        patience=7,
        verbose=True,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename=filename,
        save_top_k=1,
        verbose=True,
        monitor='val_rmspe',
        mode='min'
    )

    trainer = pl.Trainer(   logger=pl_loggers.TensorBoardLogger('./logs/'),
                            gpus=1,
                            max_epochs=100,
                            checkpoint_callback=True,
                            callbacks=[early_stop_callback,checkpoint_callback] )

    torch.manual_seed(0)
    np.random.seed(0)

    trainer.fit(model, data)

    print('time end:',datetime.now().strftime("%H:%M:%S"))


if __name__ == '__main__':

    fit_model()

