import math
from datetime import datetime
import os
import json
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from data import OptiverDataModule
from model1 import PatternFinder1
from model2 import PatternFinder2

def fit_model(CV_split):

    torch.cuda.manual_seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    pl.utilities.seed.seed_everything(0)

    # torch.backends.cudnn.benchmark = False
    # pl.utilities.seed.seed_everything(0)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

    data = OptiverDataModule(CV_split=CV_split)

    model = PatternFinder1(in_channels = data.series.shape[1])

    filename = 'optiver_CV5'+str(CV_split)
    dirpath='./checkpoints/'

    early_stop_callback = EarlyStopping(
        monitor='val_rmspe',
        patience=8,
        verbose=True,
        mode='min',
        min_delta=0.0001
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
    trainer.fit(model, data)


def eval_models(settings_path='./settings.json', device='cuda'):

    with open(settings_path) as f:
        
        settings = json.load(f)

    data = OptiverDataModule(scale=False)

    output = np.zeros(len(data.series))

    NUM_MODELS = 5

    for i in range(NUM_MODELS):

        print('model:',i)

        semf_np = np.load(settings['PREPROCESS_DIR'] + 'series_means_{}.npy'.format(i))
        sesf_np = np.load(settings['PREPROCESS_DIR'] + 'series_stds_{}.npy'.format(i))

        model = PatternFinder1.load_from_checkpoint('./checkpoints/optiver_CV5{}.ckpt'.format(i), in_channels=data.series.shape[1])

        model.to(device)
        model.eval()

        batch_size = 2**13
        num_batches = math.ceil(len(data.series) / batch_size)

        for bidx in tqdm(range(num_batches)):

            start = bidx*batch_size
            end   = start + min(batch_size, len(data.series) - start)

            if start == end:
                break

            mminput = data.series[start:end] - semf_np
            mminput = mminput / sesf_np

            output[start:end] += model(torch.tensor(mminput, dtype=torch.float32, device=device)).detach().cpu().numpy().squeeze()

    fut_rea_vol = torch.tensor(data.targets[:,3], dtype=torch.float32).to(device)
    predictions = torch.tensor(output/NUM_MODELS, dtype=torch.float32).to(device)
    evaluation = torch.sqrt(torch.mean(torch.square((fut_rea_vol - predictions) / fut_rea_vol), dim=0))
    print('result:',round(evaluation.item(),3))

    row_ids = pd.Series(data.targets[:,0].astype('int').astype('str')) + '-' + pd.Series(data.targets[:,1].astype('int').astype('str'))
    submission = pd.DataFrame({ 'row_id': row_ids, 'target': predictions.detach().cpu().numpy() })
    #submission.to_csv('submission.csv', index=False)

if __name__ == '__main__':

    # for i in range(5):
    #     print('model:',i)
    #     fit_model(CV_split=i)

    eval_models()

