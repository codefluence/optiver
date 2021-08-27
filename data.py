import os
from tqdm import tqdm
import gc


import pandas as pd
import numpy as np

# for debugging
np.set_printoptions(threshold=2000, linewidth=140, precision=3, edgeitems=20, suppress=1)
pd.set_option('display.max_rows', 601)


import torch
import torch.nn as nn

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from sklearn.preprocessing import MinMaxScaler


ROOT_DATA = 'D:/data/optiver/'


class OptiverDataModule(pl.LightningDataModule):

    def __init__(self):

        super(OptiverDataModule, self).__init__()

        epsilon = 1e-6
        kernel_size = 30
        stride = 6

        if not os.path.exists(ROOT_DATA+'cache/optiver_series.npz'):

            print('creating cached data...')
            series, targets = self.get_time_series()

            np.savez_compressed(ROOT_DATA+'cache/optiver_series.npz', series=series, targets=targets)
        else:
            print('loading cached data...')
            tensors = np.load(ROOT_DATA+'cache/optiver_series.npz')

            books_series = tensors['series']  # shape(428932, 11, 600)
            self.targets = tensors['targets']

        WAP = (books_series[:,0]*books_series[:,5] + books_series[:,1]*books_series[:,4]) / (books_series[:,4] + books_series[:,5])
        books_series[:,8] = np.nan_to_num(books_series[:,8]) + np.isnan(books_series[:,8]) * np.nan_to_num(WAP)

        #TODO: mover a get_time_series
        books_series[:,9] = np.nan_to_num(books_series[:,9])
        books_series[:,10] = np.nan_to_num(books_series[:,10])

        windows = torch.Tensor(WAP).unfold(1,kernel_size,1)
        moving_mean = torch.mean(windows, axis=2)
        moving_std  = torch.std(windows, axis=2)
        moving_min  = torch.min(windows, axis=2)
        moving_max  = torch.max(windows, axis=2)

        series = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)(torch.Tensor(books_series)).cuda()
        moving_mean = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)(moving_mean.unsqueeze(0)).squeeze().cuda()
        moving_std = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)(moving_std.unsqueeze(0)).squeeze().cuda()
        moving_min = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)(moving_min).cuda()
        moving_max = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)(moving_max).cuda()

        books_series = None
        gc.collect()

        bid_px1 = series[:,0,-91:]
        ask_px1 = series[:,1,-91:]
        bid_px2 = series[:,2,-91:]
        ask_px2 = series[:,3,-91:]
        bid_qty1 = series[:,4,-91:]
        ask_qty1 = series[:,5,-91:]
        bid_qty2 = series[:,6,-91:]
        ask_qty2 = series[:,7,-91:]
        last_wavg_px = series[:,8,-91:]
        executed_qty = series[:,9,-91:]
        executed_count = series[:,10,-91:]

        print('computing series...')

        ################################## Technical analysis ##################################

        WAP = (bid_px1*ask_qty1 + ask_px1*bid_qty1) / (bid_qty1 + ask_qty1)

        bollinger_deviation = (moving_mean.squeeze() - WAP) / (moving_std.squeeze() + epsilon)

        moving_range_diff = moving_max - moving_min  ### <----- problem
        keltner_deviation = (moving_mean.squeeze() - WAP) / (moving_range_diff.squeeze() + epsilon)

        donchian_deviation = 2 * WAP / (moving_max + moving_min) - 1

        #avg_true_range = torch.split(grouped_x[sorted_idx], chunk_size, dim=0)


        ######################################################################################################

        bOF1 = self.get_orderflow(bid_px1, bid_qty1)
        bOF2 = self.get_orderflow(bid_px2, bid_qty2)
        aOF1 = self.get_orderflow(ask_px1, ask_qty1)
        aOF2 = self.get_orderflow(ask_px2, ask_qty2)

        OFI1 = bOF1 - aOF1
        OFI2 = bOF2 - aOF2

        t_bid_size = bid_qty1 + bid_qty2
        t_ask_size = ask_qty1 + ask_qty2

        w_avg_bid_price = (bid_px1*bid_qty1 + bid_px2*bid_qty2) / t_bid_size
        w_avg_ask_price = (ask_px1*ask_qty1 + ask_px2*ask_qty2) / t_ask_size

        WAP = (bid_px1*ask_qty1 + ask_px1*bid_qty1) / (bid_qty1 + ask_qty1)
        deepWAP = (w_avg_bid_price * t_ask_size + w_avg_ask_price * t_bid_size) / (t_bid_size + t_ask_size)
        mid_price = (bid_px1 + ask_px1) / 2

        spread = ask_px1 / bid_px1 - 1  # TODO: mejor formula para el spread?
        deep_spread = w_avg_ask_price / w_avg_bid_price  -  1

        prep = torch.Tensor([0]).repeat(WAP.shape[0],1).cuda()
        log_returns = torch.diff(torch.log(WAP), prepend=prep)  # TODO: double-check prepend
        deep_log_returns = torch.diff(torch.log(deepWAP), prepend=prep)  # TODO: double-check prepend

        #VOLUME
        # vol_imbalance = (t_ask_size + epsilon) / (t_bid_size + epsilon) - 1
        # vol_sum = t_ask_size + t_bid_size 
        # # the value of a unit is different for every stock, but still the shape of the series might be useful
        # #TODO: normalizar volumenes absolutos

        ######################################## Trades ########################################

        # # #"some commonly seen technical signals of the financial market are derived from trade data directly, such as high-low or total traded volume." segun organizadores

        # last_wavg_px = np.nan_to_num(last_wavg_px) + np.isnan(last_wavg_px) * np.nan_to_num(WAP)
        # # last_log_returns = np.diff(np.log(last_wavg_px), prepend=0)  # TODO: double-check prepend
        # # log_returns_dev2 = (log_returns + epsilon) / (last_log_returns + epsilon) - 1
 
        ###############################################################################

        self.series = torch.stack((
            WAP / deepWAP - 1,
            WAP / mid_price - 1,
            # WAP / last_wavg_px - 1,
            # deepWAP / last_wavg_px - 1,
            # (log_returns + epsilon) / (deep_log_returns + epsilon) - 1,
            # (spread + epsilon) / (deep_spread + epsilon) - 1,
            WAP,
            deepWAP
        ), axis=1)

        ###############################################################################

        print('computing stats...')

        self.stats = np.repeat(np.nan, len(log_returns)).reshape(len(log_returns), 1).astype(np.float32)

        # Realized volatility
        self.stats[:,0] = np.apply_along_axis(lambda x : np.sqrt(np.sum(x**2)), 1, log_returns)   #s[:,2]
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

        #print('scaling...')

        # for i in range(NUM_SERIES):

        #     scaler = MinMaxScaler().fit(s[:,i].T)
        #     s[:,i] = scaler.transform(s[:,i].T).T

        # self.series = s

        # assert(np.isnan(self.series).sum() == 0)
        # assert(np.isinf(self.series).sum() == 0)

        # assert(np.isnan(self.stats).sum() == 0)
        # assert(np.isinf(self.stats).sum() == 0)

        print('data ready')

    def get_orderflow(self, prices, qties):

        prep = torch.Tensor([0]).repeat(qties.shape[0],1).cuda()

        vol_diff   = torch.diff(qties, prepend=prep)
        price_diff = torch.diff(prices, prepend=prep)

        vol_diff[price_diff > 0] = qties[price_diff > 0]
        vol_diff[price_diff < 0] = qties[price_diff < 0] * -1

        return vol_diff

    def train_dataloader(self):

        return DataLoader(SeriesDataSet(self.series, self.stats, self.targets[self.targets[:,0] % 5 != 0]), batch_size=1024, shuffle=True)

    def val_dataloader(self):

        return DataLoader(SeriesDataSet(self.series, self.stats, self.targets[self.targets[:,0] % 5 == 0]), batch_size=1024, shuffle=False)

    def setup(self, stage):

        pass

    def get_time_series(self):

        series = []

        targets = pd.read_csv(ROOT_DATA+'train.csv')
        targets.loc[:,'tensor_index'] = np.nan
        targets = targets.to_numpy(dtype=np.float32)

        for folder in 'book_train.parquet', 'trade_train.parquet': 

            file_paths = []
            path_root = ROOT_DATA + folder + '/'

            for path, _, files in os.walk(path_root):
                for name in files:
                    file_paths.append(os.path.join(path, name))

            for file_path in tqdm(file_paths):

                df = pd.read_parquet(file_path, engine='pyarrow')
                stock_id = int(file_path.split('\\')[-2].split('=')[-1])

                for time_id in np.unique(df.time_id):

                    df_time = df[df.time_id == time_id].reset_index(drop=True)
                    with_changes_len = len(df_time)

                    if 'book' in file_path:
                        assert df_time.seconds_in_bucket[0] == 0

                    df_time = df_time.reindex(list(range(600)))

                    missing = set(range(600)) - set(df_time.seconds_in_bucket)
                    df_time.loc[with_changes_len:,'seconds_in_bucket'] = list(missing)

                    df_time = df_time.sort_values(by='seconds_in_bucket').reset_index(drop=True)

                    if 'book' in file_path:

                        df_time = df_time.iloc[:,2:].ffill(axis=0)
                        targets[(targets[:,0]==stock_id) & (targets[:,1]==time_id), -1] = len(series)

                        series.append(np.vstack((df_time.T.to_numpy(dtype=np.float32), np.repeat(np.nan, 3*600).reshape(3,600))))

                    elif 'trade' in file_path:

                        df_time = df_time.iloc[:,2:].fillna({'size':0, 'order_count':0}).ffill(axis=0)

                        tensor_index = targets[(targets[:,0]==stock_id) & (targets[:,1]==time_id), -1].item()
                        series[int(tensor_index)][-3:] = df_time.T.to_numpy(dtype=np.float32)

        return series, targets





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

        # sliding_window_view not available for numpy < 1.20
        # moving_realized_volatility = np.apply_along_axis(lambda x: np.sqrt(np.sum(x**2)), arr=windows, axis=2)

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


