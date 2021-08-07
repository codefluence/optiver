import glob
from model import VolatilityClassifier
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch

from data import ROOT_DATA


def get_tensors(file_path):

    df = pd.read_parquet(file_path, engine='pyarrow')
    stock_id = int(file_path.split('\\')[-1].split('=')[-1])
    df['stock_id'] = stock_id

    for time_id in tqdm(np.unique(df.time_id)):

        df_time = df[df.time_id == time_id].reset_index(drop=True)
        changes_len = len(df_time)

        if 'book' in file_path:
            assert df_time.seconds_in_bucket[0] == 0  #TODO

        df_time = df_time.reindex(list(range(600))).reset_index(drop=True)

        missing = set(range(600)) - set(df_time.seconds_in_bucket)
        df_time.loc[changes_len:,'seconds_in_bucket'] = list(missing)

        df_time = df_time.sort_values(by='seconds_in_bucket').reset_index(drop=True)
        df_time.loc[:,'time_id'] = time_id

        df_time.ffill(axis = 0, inplace=True)

        yield df_time.T.to_numpy(dtype=np.float32)


if __name__ == '__main__':

    model = VolatilityClassifier.load_from_checkpoint('./weights/optiver_best.ckpt')
    model.cpu()
    model.eval()

    book_test_files = glob.glob(ROOT_DATA + '/book_val.parquet/*')

    tensors = []

    for f in book_test_files:
        print(f)
        tensors.extend(list(get_tensors(f)))

    tensors = np.stack(tensors, axis=0)

    stockids = tensors[:,-1,0]
    timeids  = tensors[:,0,0]

    df = pd.DataFrame(data=np.hstack((stockids.reshape(-1,1),timeids.reshape(-1,1))), columns=["row_id", "target"])

    df['row_id'] = df['row_id'].astype('int').astype('str') + '-' + df['target'].astype('int').astype('str')
    df['target'] = np.nan

    tensors[:,0] = (tensors[:,2]*tensors[:,7] + tensors[:,3]* tensors[:,6]) / (tensors[:,6] + tensors[:,7]) 
    tensors[:,1] = np.diff(np.log(tensors[:,0]), prepend=0)  #TODO

    stats = np.apply_along_axis(lambda x : np.sqrt(np.sum(x**2)), 1, tensors[:,1]).reshape(-1,1)

    bs = 1024

    for i in tqdm(range(len(tensors) // bs + 1)):

        sta = bs*i
        end = bs*(i+1) if bs*(i+1) < len(tensors) else len(tensors)

        if sta == end:
            break

        df.loc[np.arange(sta,end),'target'] = model(    torch.Tensor(tensors[sta:end,:-1]),
                                                        torch.Tensor(stats[sta:end])).detach().numpy().squeeze()

    df.to_csv('submission.csv', index = False)



