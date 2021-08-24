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

from sklearn.preprocessing import MinMaxScaler

ROOT_DATA = 'D:/data/optiver/'


class OptiverDataModule(pl.LightningDataModule):

    def __init__(self):

        super(OptiverDataModule, self).__init__()

        if not os.path.exists(ROOT_DATA+'cache/optiver_tensors.npz'):

            books, trades, truth = self.get_numpy_matrix()

            np.savez_compressed(ROOT_DATA+'cache/optiver_tensors_rest.npz', books=books, trades=trades, truth=truth)

        print('loading cached data...')
        tensors = np.load(ROOT_DATA+'cache/optiver_tensors.npz')

        b = tensors['books']  # (428932, 11, 600)
        self.targets = tensors['truth']

        NUM_SERIES = 8
        s = np.repeat(np.nan, b.shape[0] * NUM_SERIES * b.shape[2]).reshape(b.shape[0], NUM_SERIES, b.shape[2]).astype(np.float32)

        NUM_STATS = 9
        self.stats = np.repeat(np.nan, b.shape[0] * NUM_STATS).reshape(b.shape[0], NUM_STATS).astype(np.float32)

        epsilon = 0.00001

        bid_px1 = b[:,2]
        ask_px1 = b[:,3]
        bid_px2 = b[:,4]
        ask_px2 = b[:,5]

        bid_qty1 = b[:,6]
        ask_qty1 = b[:,7]
        bid_qty2 = b[:,8]
        ask_qty2 = b[:,9]

        t_bid_size = bid_qty1 + bid_qty2
        t_ask_size = ask_qty1 + ask_qty2

        w_avg_bid_price = (bid_px1*bid_qty1 + bid_px2*bid_qty2) / t_bid_size
        w_avg_ask_price = (ask_px1*ask_qty1 + ask_px2*ask_qty2) / t_ask_size

        print('computing series...')

        WAP = (bid_px1*ask_qty1 + ask_px1*bid_qty1) / (bid_qty1 + ask_qty1)

        spread = ask_px1 / bid_px1 - 1  # TODO: mejor formula para el spread?

        log_returns = np.diff(np.log(WAP), prepend=0)  # TODO: double-check prepend

        # deepSpread: Alternative to spread that takes into account the second level
        deep_spread = w_avg_ask_price / w_avg_bid_price  -  1

        deepWAP = (w_avg_bid_price * t_ask_size + w_avg_ask_price * t_bid_size) / (t_bid_size + t_ask_size)

        deep_log_returns = np.diff(np.log(deepWAP), prepend=0)  # TODO: double-check prepend

        #TODO: diff between spread and deepSpread, etc en lugar de lo anterior?
        WAPdev = WAP / deepWAP - 1
        log_returns_dev = (log_returns + epsilon) / (deep_log_returns + epsilon) - 1
        spread_dev = (spread + epsilon) / (deep_spread + epsilon) - 1

        vol_imbalance = (t_ask_size + epsilon) / (t_bid_size + epsilon) - 1

        vol_sum = t_ask_size + t_bid_size 
        # the value of a unit is different for every stock, but still the shape of the series might be useful
        #TODO: normalizar volumenes absolutos

        # sliding_window_view not available for numpy < 1.20
        windows = torch.Tensor(log_returns).unfold(1,30,6).detach().numpy()
        moving_realized_volatility = np.std(windows, axis=1)
        # moving_realized_volatility = np.apply_along_axis(lambda x: np.sqrt(np.sum(x**2)), arr=windows, axis=2)

        ###############################################################################

        t = tensors['trades']

        last_wavg_px = t[:,2]
        executed_qty = t[:,3]
        executed_count = t[:,4]

        # At the beginning of the window there is no "last execution" carried from previous time, we use WAP instead 
        last_wavg_px = np.nan_to_num(last_wavg_px) + np.isnan(last_wavg_px) * np.nan_to_num(WAP)

        s[:,4] = WAP / last_wavg_px - 1
        s[:,4] = deepWAP / last_wavg_px - 1

        last_log_returns = np.diff(np.log(last_wavg_px), prepend=0)  # TODO: double-check prepend

        log_returns_dev2 = (log_returns + epsilon) / (last_log_returns + epsilon) - 1

        s = np.vstack((WAP,spread,log_returns,deep_spread,deepWAP,deep_log_returns,WAPdev,
                       log_returns_dev,spread_dev,vol_imbalance,vol_sum,moving_realized_volatility,
                       last_wavg_px,last_log_returns,log_returns_dev2))
   
        ###############################################################################

        # technical analysis



        ###############################################################################

        print('computing stats...')
        # Realized volatility
        self.stats[:,0] = np.apply_along_axis(lambda x : np.sqrt(np.sum(x**2)), 1, s[:,2])
        # self.stats[:,1] = np.apply_along_axis(lambda x : np.sqrt(np.sum(x**2)), 1, s[:,2,-300:])
        # self.stats[:,2] = np.apply_along_axis(lambda x : np.sqrt(np.sum(x**2)), 1, s[:,2,-100:])
        #TODO: repetir con trades

        # # Realized volatility trend
        # self.stats[:,3] = np.apply_along_axis(lambda x : np.sqrt(np.sum(x[-300:]**2)) - np.sqrt(np.sum(x[:300]**2)), 1, s[:,2])
        # self.stats[:,4] = np.apply_along_axis(lambda x : np.sqrt(np.sum(x[-150:]**2)) - np.sqrt(np.sum(x[-300:-150]**2)), 1, s[:,2])
        # self.stats[:,5] = np.apply_along_axis(lambda x : np.sqrt(np.sum(x[-50:]**2)) - np.sqrt(np.sum(x[-100:-50]**2)), 1, s[:,2])

        # # Spread trend
        # self.stats[:,6] = np.apply_along_axis(lambda x : np.mean(x[-300:]) - np.mean(x[:300]), 1, s[:,1])
        # self.stats[:,7] = np.apply_along_axis(lambda x : np.mean(x[-150:]) - np.mean(x[-300:-150]), 1, s[:,1])
        # self.stats[:,8] = np.apply_along_axis(lambda x : np.mean(x[-50:]) - np.mean(x[-100:-50]), 1, s[:,1])

        #TODO: normalize stats?

        #TODO: 'sum', 'mean', 'std', 'max', 'min'

        ###############################################################################

        print('scaling...')

        # for i in range(NUM_SERIES):

        #     scaler = MinMaxScaler().fit(s[:,i].T)
        #     s[:,i] = scaler.transform(s[:,i].T).T

        self.series = s

        assert(np.isnan(self.series).sum() == 0)
        assert(np.isinf(self.series).sum() == 0)

        # assert(np.isnan(self.stats).sum() == 0)
        # assert(np.isinf(self.stats).sum() == 0)

        print('data ready')

    def train_dataloader(self):

        return DataLoader(SeriesDataSet(self.series, self.stats, self.targets[self.targets[:,0] % 5 != 0]), batch_size=1024, shuffle=True)

    def val_dataloader(self):

        return DataLoader(SeriesDataSet(self.series, self.stats, self.targets[self.targets[:,0] % 5 == 0]), batch_size=1024, shuffle=False)

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





class SeriesDataSet(Dataset):
    
    def __init__(self, series, stats, targets):

        super(SeriesDataSet, self).__init__()

        self.series = series
        self.stats = stats
        self.targets = targets

    def __len__(self):

        return len(self.targets)

    def __getitem__(self, idx):

        series_index = int(self.targets[idx,-2])

        return self.series[series_index], self.stats[series_index], self.targets[idx,-3]



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


