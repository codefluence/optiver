import os
import gc
import json
import math
from tqdm import tqdm

import pandas as pd
import numpy as np

import torch
import torch.nn as nn

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from sklearn.preprocessing import MinMaxScaler

np.set_printoptions(threshold=2000, linewidth=140, precision=3, edgeitems=20, suppress=1)
pd.set_option('display.max_rows', 601)



class OptiverDataModule(pl.LightningDataModule):

    def __init__(self):

        super(OptiverDataModule, self).__init__()

        with open('./settings.json') as f:
            self.settings = json.load(f)

        kernel_size = 30
        stride = 6
        epsilon = 1e-6

        if not os.path.exists(self.settings['ROOT_DATA'] +'cache/optiver_series.npz'):

            print('creating cached data...')
            series, targets = self.get_time_series()

            np.savez_compressed(self.settings['ROOT_DATA'] + 'cache/optiver_series.npz', series=series, targets=targets)

        print('loading cached data...')
        tensors = np.load(self.settings['ROOT_DATA'] + 'cache/optiver_series.npz')

        self.targets = tensors['targets'].astype(np.float32)

        #TODO: get_time_series debe guardar en np.float32
        series = tensors['series'].astype(np.float32)  # shape(428932, 11, 600)
        #TODO: mover a get_time_series:
        series[:,9] = np.nan_to_num(series[:,9])
        series[:,10] = np.nan_to_num(series[:,10])

        bid_px1 = series[:,0]
        ask_px1 = series[:,1]
        bid_px2 = series[:,2]
        ask_px2 = series[:,3]

        bid_qty1 = series[:,4]
        ask_qty1 = series[:,5]
        bid_qty2 = series[:,6]
        ask_qty2 = series[:,7]

        executed_px = series[:,8]
        executed_qty = series[:,9]
        executed_count = series[:,10]

        series = None

        print('processing series...')

        sumexecs = self.targets[:,:2].copy()
        sumexecs[:,1] = np.sum(executed_qty, axis=1)

        sumexecs_means = sumexecs.copy()

        for i in np.unique(sumexecs[:,0]):
            sumexecs_means[sumexecs_means[:,0]==int(i),1] = np.mean(sumexecs[sumexecs[:,0]==int(i)][:,1])

        sumexecs = sumexecs[:,1:2]
        sumexecs_means = sumexecs_means[:,1:2]

        def reduce(array, log=''):

            a = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)(torch.Tensor(array).cuda().unsqueeze(0)).squeeze().cpu().numpy()

            copy = a.copy()

            del a
            torch.cuda.empty_cache()

            assert(np.isnan(copy).sum() == 0)
            assert(np.isinf(copy).sum() == 0)

            print(log, '\tmin:', np.round(np.min(copy),5), '\tmax:', np.round(np.max(copy),5))

            return copy

        # executed_qty_dist =  reduce(np.divide(executed_qty, sumexecs))
        # executed_count_dist = reduce(np.divide(executed_count, np.expand_dims(np.sum(executed_count,axis=1),axis=1)))
        executed_count = None

        def get_orderflow(prices, qties):

            torch.cuda.empty_cache()

            prices = torch.Tensor(prices).cuda()
            qties = torch.Tensor(qties / sumexecs_means).cuda()

            price_diff = torch.diff(prices)
            vol_diff   = torch.diff(qties)

            of = (price_diff == 0) * vol_diff + \
                 (price_diff > 0)  * qties[:,1:] + \
                 (price_diff < 0)  * torch.roll(qties * -1, 1, 1)[:,1:]

            return of.cpu().numpy()

        OFI1 = reduce(get_orderflow(bid_px1, bid_qty1) - get_orderflow(ask_px1, ask_qty1), 'OFI1   ')
        OFI2 = reduce(get_orderflow(bid_px2, bid_qty2) - get_orderflow(ask_px2, ask_qty2), 'OFI2   ')

        torch.cuda.empty_cache()

        bid_px2 = reduce(bid_px2, 'bid_px2')
        ask_px2 = reduce(ask_px2, 'ask_px2')
        bid_qty2 = reduce(bid_qty2, 'bid_qty2')
        ask_qty2 = reduce(ask_qty2, 'ask_qty2')

        gc.collect()

        WAP = (bid_px1*ask_qty1 + ask_px1*bid_qty1) / (bid_qty1 + ask_qty1)
        executed_px = np.nan_to_num(executed_px) + np.isnan(executed_px) * np.nan_to_num(WAP)
        
        log_returns = np.diff(np.log(WAP))
        log_returns = reduce(log_returns, 'log_returns')

        bid_px1 = reduce(bid_px1, 'bid_px1')
        ask_px1 = reduce(ask_px1, 'ask_px1')
        bid_qty1 = reduce(bid_qty1, 'bid_qty1')
        ask_qty1 = reduce(ask_qty1, 'ask_qty1')

        gc.collect()
        torch.cuda.empty_cache()

        ################################ Inspired on Technical Analysis ################################

        # Inspired on https://www.investopedia.com/terms/o/onbalancevolume.asp
        OBV = ((executed_qty[:,1:] * (np.diff(executed_px) > 0)) - (executed_qty[:,1:] * (np.diff(executed_px) < 0))) / sumexecs_means
        OBV = reduce(OBV, 'OBV    ')

        # Inspired on https://www.investopedia.com/terms/f/force-index.asp
        force_index = np.diff(np.log(executed_px)) * executed_qty[:,1:] * 1e5 / sumexecs_means
        force_index = reduce(force_index, 'force_ix')

        # TODO: ??????????
        # executed_qty_windows = torch.Tensor(executed_qty).cuda().unfold(1,kernel_size,1)
        # moving_executed_qty = torch.mean(executed_qty_windows, axis=2)
        # effort = (executed_qty[:,1:] / log_returns) / moving_executed_qty
        # effort = torch.Tensor(np.nan_to_num(effort))
        # effort = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)(effort.unsqueeze(0)).squeeze().cpu()


        # Inspired on https://www.investopedia.com/terms/m/mfi.asp
        
        money_flow = (executed_px * executed_qty)[:,1:]

        money_flow_pos = torch.Tensor(money_flow * (np.diff(executed_px) > 0)).cuda()
        money_flow_neg = torch.Tensor(money_flow * (np.diff(executed_px) < 0)).cuda()

        # unfold generates a view, to save GPU memory it's better to move to cuda the original tensor before unfolding 
        money_flow_pos_windows = money_flow_pos.unfold(1,kernel_size,1)
        money_flow_neg_windows = money_flow_neg.unfold(1,kernel_size,1)

        money_flow_pos_sums = torch.nansum(money_flow_pos_windows, axis=2)
        money_flow_neg_sums = torch.nansum(money_flow_neg_windows, axis=2)

        MFI = money_flow_pos_sums / (money_flow_pos_sums + money_flow_neg_sums)
        MFI = reduce(torch.nan_to_num(MFI,nan=0.5).cpu().numpy(), 'MFI    ')

        #TODO: rengo que hacer /cpu()?
        del money_flow_pos
        del money_flow_neg
        del money_flow_pos_windows
        del money_flow_neg_windows
        del money_flow_pos_sums
        del money_flow_neg_sums

        gc.collect()
        torch.cuda.empty_cache()


        # Inspired on https://www.investopedia.com/terms/v/vwap.asp

        money_flows_windows = torch.Tensor(money_flow).cuda().unfold(1,kernel_size,1)
        executed_qty_windows = torch.Tensor(executed_qty[:,1:]).cuda().unfold(1,kernel_size,1)

        VWAP = torch.sum(money_flows_windows, axis=2) / torch.sum(executed_qty_windows, axis=2)
        VWAP = pd.DataFrame(VWAP.cpu().numpy()).ffill(axis=1).values
        VWAP = np.nan_to_num(VWAP) + np.isnan(VWAP) * np.nan_to_num(WAP[:,:VWAP.shape[1]])  # beginning of series
        VWAP = reduce(VWAP, 'VWAP    ')

        del money_flows_windows
        del executed_qty_windows
        torch.cuda.empty_cache()


        ####

        batch_size = 100000
        num_batches = math.ceil(len(WAP) / batch_size)

        moving_mean = []
        moving_std = []
        moving_min = []
        moving_max = []

        for bidx in range(num_batches):

            start = int(bidx*batch_size)
            end = int(bidx*batch_size) + min(batch_size, len(WAP) - start)

            WAP_windows = torch.Tensor(WAP[start:end]).cuda().unfold(1,kernel_size,1)

            moving_mean.append(torch.mean(WAP_windows, axis=2).cpu())  # .copy()
            moving_std.append(torch.std(WAP_windows, axis=2).cpu())
            moving_min.append(torch.min(WAP_windows, axis=2).values.cpu())
            moving_max.append(torch.max(WAP_windows, axis=2).values.cpu())

            torch.cuda.empty_cache()


        # Inspired on https://www.investopedia.com/terms/u/.....

        moving_mean = torch.vstack(moving_mean).cuda()
        moving_std = torch.vstack(moving_std).cuda()

        WAP_cuda = torch.Tensor(WAP[:,-571:]).cuda()
        bollinger_deviation = reduce(((moving_mean.cuda() - WAP_cuda) / (moving_std.cuda() + epsilon)).cpu().numpy())  #TODO moving_std==0 case

        moving_mean = reduce(moving_mean.cpu().numpy())
        moving_std = reduce(moving_std.cpu().numpy())

        moving_min = torch.Tensor(np.vstack(moving_min)).cuda()
        moving_max = torch.Tensor(np.vstack(moving_max)).cuda()

        gc.collect()
        torch.cuda.empty_cache()


        # Inspired on https://www.investopedia.com/terms/u/ulcerindex.asp

        R = reduce(torch.pow(100 * (WAP_cuda - moving_max) / moving_max, 2).cpu().numpy(), 'R       ')

        del WAP_cuda
        torch.cuda.empty_cache()


        # Inspired on https://www.investopedia.com/articles/trading/08/accumulation-distribution-line.asp
        lexpx = torch.Tensor(executed_px[:,-571:]).cuda()
        CLV = torch.nan_to_num(((lexpx - moving_min) - (moving_max - lexpx)) / (moving_max - moving_min), nan=0., posinf=0, neginf=0)

        del lexpx
        torch.cuda.empty_cache()

        moving_min = reduce(moving_min.cpu().numpy(), 'mov_min')
        moving_max = reduce(moving_max.cpu().numpy(), 'mov_max')


        # Inspired on https://www.investopedia.com/terms/c/chaikinoscillator.asp
        lexqty = torch.Tensor(executed_qty[:,-571:]).cuda()
        execlv_windows = (lexqty*CLV).unfold(1,kernel_size,1)
        vol_windows = lexqty.unfold(1,kernel_size,1)
        CMF = torch.nan_to_num(torch.mean(execlv_windows, axis=2) / torch.mean(vol_windows, axis=2), nan=0., posinf=0, neginf=0)

        del lexqty
        del execlv_windows
        del vol_windows
        torch.cuda.empty_cache()

        CLV = reduce(CLV.cpu().numpy(), 'CLV    ')
        CMF = reduce(CMF.cpu().numpy(), 'CMF    ')

        #571



        # ATR = moving_max / moving_min - 1

        # middle_channel = (moving_max + moving_min) / 2
        # donchian_deviation = (middle_channel - WAP) / (moving_max - moving_min + epsilon)  #TODO

        ################################################################################################

        WAP = reduce(WAP, 'WAP    ')
        executed_px = reduce(executed_px, 'exe_px')
        executed_qty = reduce(executed_qty, 'exe_qty')

        gc.collect()

        t_bid_size = bid_qty1 + bid_qty2
        t_ask_size = ask_qty1 + ask_qty2

        vol = (t_bid_size + t_ask_size) / (torch.Tensor(sumexecs_means).cuda() + epsilon)

        vol_unbalance1 = t_ask_size / ( t_ask_size + t_bid_size + epsilon)
        vol_unbalance2 = (ask_qty2 + bid_qty2) / (ask_qty1 + bid_qty1 + epsilon)
        vol_unbalance3 = (ask_qty1 + bid_qty2) / (ask_qty2 + bid_qty1 + epsilon)

        w_avg_bid_price = (bid_px1*bid_qty1 + bid_px2*bid_qty2) / t_bid_size
        w_avg_ask_price = (ask_px1*ask_qty1 + ask_px2*ask_qty2) / t_ask_size

        deepWAP = (w_avg_bid_price * t_ask_size + w_avg_ask_price * t_bid_size) / (t_bid_size + t_ask_size)
        mid_price = (bid_px1 + ask_px1) / 2

        spread = ask_px1 / bid_px1 - 1  # TODO: mejor formula para el spread?
        deep_spread = w_avg_ask_price / w_avg_bid_price  -  1

        deep_log_returns = torch.diff(torch.log(deepWAP))


        ######################################## Trades ########################################

        # # #"some commonly seen technical signals of the financial market are derived from trade data directly, such as high-low or total traded volume." segun organizadores

        # last_wavg_px = np.nan_to_num(last_wavg_px) + np.isnan(last_wavg_px) * np.nan_to_num(WAP)
        # # last_log_returns = np.diff(np.log(last_wavg_px), prepend=0)  # TODO: double-check prepend
        # # log_returns_dev2 = (log_returns + epsilon) / (last_log_returns + epsilon) - 1

        ###############################################################################

        # TODOS: derivadas (diff), visualizar para ver si se puede mejorar algo

        nels = 91

        self.series = torch.stack((
            moving_std[:,-nels:],
            VWAP[:,-nels:],
            MFI[:,-nels:],
            log_returns[:,-nels:],
            spread[:,-nels:],
            OFI1[:,-nels:],
            OFI2[:,-nels:],
            vol[:,-nels:],
            vol_unbalance1[:,-nels:],
            vol_unbalance2[:,-nels:],
            vol_unbalance3[:,-nels:],
            (WAP / deepWAP - 1)[:,-nels:],
            (WAP / mid_price - 1)[:,-nels:],
            (WAP / executed_px - 1)[:,-nels:],
            (deepWAP / executed_px - 1)[:,-nels:],
            ((log_returns + epsilon) / (deep_log_returns + epsilon) - 1)[:,-nels:],
            ((spread + epsilon) / (deep_spread + epsilon) - 1)[:,-nels:]
        ), axis=1)

        ###############################################################################

        print('computing stats...')

        self.stats = np.repeat(np.nan, len(log_returns)).reshape(len(log_returns), 1).astype(np.float32)

        # Realized volatility
        # self.stats[:,0] = np.apply_along_axis(lambda x : np.sqrt(np.sum(x**2)), 1, log_returns)   #s[:,2]
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

        ###############################################################################

        #print('scaling...')

        # for i in range(NUM_SERIES):

        #     scaler = MinMaxScaler().fit(s[:,i].T)
        #     s[:,i] = scaler.transform(s[:,i].T).T

        self.series = self.series.cpu()
        self.series = (self.series - torch.mean(self.series, dim=(0,2)).unsqueeze(1)) / torch.std(self.series, dim=(0,2)).unsqueeze(1)
        self.series = self.series.cuda()

        # self.series = s

        # assert(np.isnan(self.series).sum() == 0)
        # assert(np.isinf(self.series).sum() == 0)

        # assert(np.isnan(self.stats).sum() == 0)
        # assert(np.isinf(self.stats).sum() == 0)

        print('data ready')

    def train_dataloader(self):

        return DataLoader(SeriesDataSet(self.series, self.stats, self.targets[self.targets[:,0] % 5 != 0]), batch_size=1024, shuffle=True)

    def val_dataloader(self):

        return DataLoader(SeriesDataSet(self.series, self.stats, self.targets[self.targets[:,0] % 5 == 0]), batch_size=1024, shuffle=False)

    # def train_dataloader(self):

    #     return DataLoader(SeriesDataSet(self.series, self.stats, self.targets[self.targets[:,0] < 2]), batch_size=1024, shuffle=True)

    # def val_dataloader(self):

    #     return DataLoader(SeriesDataSet(self.series, self.stats, self.targets[self.targets[:,0] == 2]), batch_size=1024, shuffle=False)


    def setup(self, stage):

        pass

    def get_time_series(self):

        series = []

        targets = pd.read_csv(self.settings['ROOT_DATA'] + 'train.csv')
        targets.loc[:,'tensor_index'] = np.nan
        targets = targets.to_numpy(dtype=np.float32)

        for folder in 'book_train.parquet', 'trade_train.parquet': 

            file_paths = []
            path_root = self.settings['ROOT_DATA'] + folder + '/'

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

        series_index = int(self.targets[idx,-1])

        return self.series[series_index], self.stats[series_index], self.targets[idx,-2]



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


