import math
from datetime import datetime

import torch
import numpy as np
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from data import OptiverDataModule
from model import VolatilityClassifier, PatternFinder

from net1d import Net1D
from resnet1d import ResNet1D

def fit_model(CV_split):

    data = OptiverDataModule(CV_split=CV_split)

    model = PatternFinder(in_channels=data.series.shape[1])
    #model = VolatilityClassifier(data.stats.shape[1])
    # model = ResNet1D(
    #                     in_channels=15, 
    #                     base_filters=15, 
    #                     kernel_size=3, 
    #                     stride=2, 
    #                     n_block=4, 
    #                     groups=3,
    #                     n_classes=1, 
    #                     downsample_gap=max(3//8, 1), 
    #                     increasefilter_gap=max(3//4, 1), 
    #                     verbose=False)

    filename = 'optiver_CV5'+str(CV_split)
    dirpath='./checkpoints/'

    early_stop_callback = EarlyStopping(
        monitor='val_monit',
        patience=8,
        verbose=True,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename=filename,
        save_top_k=1,
        verbose=True,
        monitor='val_monit',
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


def eval_model(CV_split, device='cuda'):

    data = OptiverDataModule()

    model = PatternFinder.load_from_checkpoint('./checkpoints/optiver_CV5{}.ckpt'.format(CV_split), in_channels=data.series.shape[1])

    model.cuda()
    model.eval()

    output = np.zeros(len(data.series))
    output[:] = np.nan

    batch_size = 2**14
    num_batches = math.ceil(len(data.series) / batch_size)

    for bidx in range(num_batches):

        start = bidx*batch_size
        end   = start + min(batch_size, len(data.series) - start)
        print(start,end)

        if start == end:
            break

        output[start:end]= model(torch.tensor(data.series[start:end], dtype=torch.float32, device=device))[0].detach().cpu().numpy().squeeze()

    targets = torch.tensor(data.targets[:,-1], dtype=torch.float32).cuda()
    predictions = torch.tensor(output, dtype=torch.float32).cuda()
    evaluation = torch.sqrt(torch.mean(torch.square((targets - predictions) / targets), dim=0))
    print(evaluation)

    row_ids = pd.Series(data.targets[:,0].astype('int').astype('str')) + '-' + pd.Series(data.targets[:,1].astype('int').astype('str'))
    submission = pd.DataFrame({ 'row_id': row_ids, 'target': predictions.detach().cpu().numpy() })
    #submission.to_csv('submission.csv', index=False)

if __name__ == '__main__':

    fit_model(0)

