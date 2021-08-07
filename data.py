import os
from tqdm import tqdm

import pandas as pd
import numpy as np

# for debugging
np.set_printoptions(threshold=2000, linewidth=140, precision=3, edgeitems=20, suppress=1)
pd.set_option('display.max_rows', 601)


import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT_DATA = 'D:/data/optiver/'


# some commonly seen technical signals of the financial market are derived from trade data directly, such as high-low or total traded volume.

class OptiverDataModule(pl.LightningDataModule):

    def __init__(self):

        super(OptiverDataModule, self).__init__()

        if not os.path.exists(ROOT_DATA+'cache/optiver_tensors.npz'):

            books, trades, truth = self.get_numpy_matrix()

            np.savez_compressed(ROOT_DATA+'cache/optiver_tensors_rest.npz', books=books, trades=trades, truth=truth)

        tensors = np.load(ROOT_DATA+'cache/optiver_tensors.npz')

        self.books = tensors['books']
        self.trades = tensors['trades']
        self.truth = tensors['truth']

        #WAP (2*7 + 3*6) / (6+7)
        self.books[:,0] = (self.books[:,2]*self.books[:,7] + self.books[:,3]* self.books[:,6]) / (self.books[:,6] + self.books[:,7]) 

        #returns
        self.books[:,1] = np.diff(np.log(self.books[:,0]), prepend=0)  #TODO
    
        self.stats = np.apply_along_axis(lambda x : np.sqrt(np.sum(x**2)), 1, self.books[:,1])

        pass

        # self.series[:,0,:] = (self.series[:,3]*self.series[:,7] + self.series[:,5]* self.series[:,9]) / \
        #                      (self.series[:,7] + self.series[:,9]) - \
        #                      (self.series[:,2]*self.series[:,6] + self.series[:,6]* self.series[:,8]) / \
        #                      (self.series[:,6] + self.series[:,8])

        #self.truth = self.truth[self.truth[:,0]==0]

    def log_return(self, list_stock_prices):

        return np.log(list_stock_prices).diff() 

    def train_dataloader(self):

        return DataLoader(SeriesDataSet(self.books, self.stats, self.truth[self.truth[:,0]<110]), batch_size=1024, shuffle=True)

    def val_dataloader(self):

        return DataLoader(SeriesDataSet(self.books, self.stats, self.truth[self.truth[:,0]>110]), batch_size=1024, shuffle=False)

    def setup(self, stage):

        pass

    def get_numpy_matrix(self):

        truth = pd.read_csv(ROOT_DATA+'train.csv').to_numpy()
        newcols = np.repeat(np.nan,truth.shape[0]*2).reshape(truth.shape[0],2)
        targets = np.hstack((truth, newcols)).astype(np.float32)

        for folder in 'book_train.parquet', 'trade_train.parquet': 

            file_paths = []
            series = []

            path_root = ROOT_DATA + folder + '/'

            for path, _, files in os.walk(path_root):
                for name in files:
                    file_paths.append(os.path.join(path, name))

            for file_path in tqdm(file_paths):

                df = pd.read_parquet(file_path, engine='pyarrow')
                stock_id = int(file_path.split('\\')[-2].split('=')[-1])
                df['stock_id'] = stock_id

                for time_id in np.unique(df.time_id):

                    df_time = df[df.time_id == time_id].reset_index(drop=True)
                    changes_len = len(df_time)

                    if 'book' in file_path:
                        assert df_time.seconds_in_bucket[0] == 0

                    df_time = df_time.reindex(list(range(600))).reset_index(drop=True)

                    missing = set(range(600)) - set(df_time.seconds_in_bucket)
                    df_time.loc[changes_len:,'seconds_in_bucket'] = list(missing)

                    df_time = df_time.sort_values(by='seconds_in_bucket').reset_index(drop=True)
                    df_time.loc[:,'time_id'] = time_id

                    if 'book' in file_path:

                        df_time.ffill(axis = 0, inplace=True)
                        target_col = -2

                    elif 'trade' in file_path:

                        df_time.loc[:,'stock_id'] = stock_id
                        df_time.fillna({'size':0, 'order_count':0}, inplace=True)
                        df_time.ffill(axis=0, inplace=True)
                        # df_time.bfill(axis=0, inplace=True)
                        target_col = -1

                    else:

                        raise Exception('Unknown fill method')

                    targets[(targets[:,0]==stock_id) & (targets[:,1]==time_id), target_col] = len(series)

                    series.append(df_time.T.to_numpy(dtype=np.float32))

            yield series

        return targets

    def get_numpy_matrix2(self):

        truth = pd.read_csv(ROOT_DATA+'train.csv').to_numpy()
        newcols = np.repeat(np.nan,truth.shape[0]*2).reshape(truth.shape[0],2)
        targets = np.hstack((truth, newcols)).astype(np.float32)

        for folder in 'book_train.parquet', 'trade_train.parquet': 

            file_paths = []
            series = []

            path_root = ROOT_DATA + folder + '/'

            for path, _, files in os.walk(path_root):
                for name in files:
                    file_paths.append(os.path.join(path, name))

            for file_path in tqdm(file_paths):

                df = pd.read_parquet(file_path, engine='pyarrow').to_numpy(dtype=np.float32)
                stock_id = int(file_path.split('\\')[-2].split('=')[-1])

                for time_id in df[:,0]:

                    w = np.empty((600,df.shape[1]), dtype=np.float32)
                    
                    w[:] = np.nan

                    subdf = df[df[:,0]==time_id]

                    w[subdf[:,1].astype(int)] = subdf

                    if 'book' in file_path:

                        assert w[0,1] == 0

                        w[:,1] = time_id
                        w[:,0] = stock_id

                        self.numpy_fill(w)

                        truth_ref_col = -2

                    elif 'trade' in file_path:

                        w[:,1] = time_id
                        w[:,0] = stock_id

                        # df_time.fillna({'size':0, 'order_count':0}, inplace=True)
                        np.nan_to_num(w, copy=False)
                        self.numpy_fill(w)

                        # df_time.bfill(axis=0, inplace=True)
                        truth_ref_col = -1

                    else:

                        raise Exception('Unknown object')

                    targets[(targets[:,0]==stock_id) & (targets[:,1]==time_id), truth_ref_col] = len(series)
                    
                    series.append(w.T)

            yield series

        yield targets

    def numpy_fill(self, arr):

        '''https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array'''

        mask = np.isnan(arr.T)
        idx = np.where(~mask,np.arange(mask.shape[1]),0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        arr.T[mask] = arr.T[np.nonzero(mask)[0], idx[mask]]



class SeriesDataSet(Dataset):
    
    def __init__(self, books, stats, truth):

        super(SeriesDataSet, self).__init__()

        self.books = books
        self.stats = stats
        self.truth = truth

    def __len__(self):

        return len(self.truth)

    def __getitem__(self, idx):

        books_index = int(self.truth[idx,-2])

        return self.books[books_index,:-1], self.stats[books_index], self.truth[idx,-3]



if __name__ == '__main__':

    data = OptiverDataModule()

    # truth = pd.read_csv(ROOT_DATA+'train.csv')

    # stockids = np.unique(truth.stock_id)
    # timeids  = np.unique(truth.time_id)

    # thematrix = np.empty((max(stockids)+1, max(timeids)+1))
    # thematrix[:] = np.nan

    # for sid in stockids:

    #     subdf = truth.loc[truth.stock_id == sid]
    #     thematrix[sid,subdf.time_id] = subdf.target

    # arma = np.nanargmax(thematrix)
    # thematrix[arma // thematrix.shape[1], arma % thematrix.shape[1]]


    pass


