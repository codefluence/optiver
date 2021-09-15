import os
import gc
import json
import math
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

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

        self.targets = tensors['targets']

        series = tensors['series']  # shape(428932, 11, 600)

        #TODO: recrear tensores file:
        series[:,9] = np.nan_to_num(series[:,9])
        series[:,10] = np.nan_to_num(series[:,10])



        print('processing series...')

        sumexecs = self.targets[:,:2].copy()
        sumexecs[:,1] = np.sum(series[:,9], axis=1)

        sumexecs_means = sumexecs.copy()

        for i in np.unique(sumexecs[:,0]):
            idx = sumexecs[:,0] == int(i)
            sumexecs_means[idx,1] = np.mean(sumexecs[idx][:,1])

        sumexecs_means = sumexecs_means[:,1:2]
        del sumexecs

        def reduce(array, log=''):

            a = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)(array.unsqueeze(0)).squeeze().cpu().numpy()

            copy = a.copy()

            del a
            torch.cuda.empty_cache()

            assert(np.isnan(copy).sum() == 0)
            assert(np.isinf(copy).sum() == 0)

            #print(log, '\tmin:', np.round(np.min(copy),5), '\tmax:', np.round(np.max(copy),5))

            return copy

        def get_orderflow(prices, qties):

            torch.cuda.empty_cache()

            price_diff = torch.diff(prices)
            vol_diff   = torch.diff(qties)

            return (price_diff == 0) * vol_diff + \
                   (price_diff > 0)  * qties[:,1:] - \
                   (price_diff < 0)  * torch.roll(qties, 1, 1)[:,1:]



        device = 'cuda'

        processed = []

        batch_size = 50000
        num_batches = math.ceil(len(series) / batch_size)

        for bidx in range(num_batches):

            start = int(bidx*batch_size)
            end = int(bidx*batch_size) + min(batch_size, len(series) - start)

            bid_px1 = torch.tensor(series[start:end,0], device=device)
            ask_px1 = torch.tensor(series[start:end,1], device=device)
            bid_px2 = torch.tensor(series[start:end,2], device=device)
            ask_px2 = torch.tensor(series[start:end,3], device=device)

            bid_qty1 = torch.tensor(series[start:end,4], device=device)
            ask_qty1 = torch.tensor(series[start:end,5], device=device)
            bid_qty2 = torch.tensor(series[start:end,6], device=device)
            ask_qty2 = torch.tensor(series[start:end,7], device=device)

            executed_px = torch.tensor(series[start:end,8], device=device)
            executed_qty = torch.tensor(series[start:end,9], device=device)
            #executed_count = torch.tensor(series[start:end,10], device=device)

            # executed_qty_dist =  reduce(np.divide(executed_qty, sumexecs))
            # executed_count_dist = reduce(np.divide(executed_count, np.expand_dims(np.sum(executed_count,axis=1),axis=1)))

            stock_means = torch.tensor(sumexecs_means[start:end], device=device)

            OFI1 = reduce(get_orderflow(bid_px1, bid_qty1 / stock_means,) - get_orderflow(ask_px1, ask_qty1 / stock_means,), 'OFI1   ')
            OFI2 = reduce(get_orderflow(bid_px2, bid_qty2 / stock_means,) - get_orderflow(ask_px2, ask_qty2 / stock_means,), 'OFI2   ')

            torch.cuda.empty_cache()

            t_bid_size = bid_qty1 + bid_qty2
            t_ask_size = ask_qty1 + ask_qty2

            vol_total = reduce((t_bid_size + t_ask_size) / (stock_means + epsilon))

            vol_unbalance1 = reduce(t_ask_size / ( t_ask_size + t_bid_size + epsilon))
            vol_unbalance2 = reduce((ask_qty2 + bid_qty2) / (ask_qty1 + bid_qty1 + epsilon))
            vol_unbalance3 = reduce((ask_qty1 + bid_qty2) / (ask_qty2 + bid_qty1 + epsilon))

            # vol_total = reduce(torch.diff(torch.log(t_bid_size + t_ask_size)))

            # vol_unbalance1 = reduce(t_ask_size / ( t_ask_size + t_bid_size + epsilon))
            # vol_unbalance2 = reduce(torch.diff((ask_qty2 + bid_qty2) / (ask_qty1 + bid_qty1 + epsilon)))
            # vol_unbalance3 = reduce(torch.diff((ask_qty1 + bid_qty2) / (ask_qty2 + bid_qty1 + epsilon)))

            w_avg_bid_price = (bid_px1*bid_qty1 + bid_px2*bid_qty2) / t_bid_size
            w_avg_ask_price = (ask_px1*ask_qty1 + ask_px2*ask_qty2) / t_ask_size

            del bid_px2
            del ask_px2
            del bid_qty2
            del ask_qty2
            torch.cuda.empty_cache()

            WAP = (bid_px1*ask_qty1 + ask_px1*bid_qty1) / (bid_qty1 + ask_qty1)

            deepWAP = (w_avg_bid_price * t_ask_size + w_avg_ask_price * t_bid_size) / (t_bid_size + t_ask_size)
            deepWAPtoWAP = reduce(deepWAP / WAP -1)

            del deepWAP
            del t_bid_size
            del t_ask_size
            torch.cuda.empty_cache()
 
            #deep_spread =  reduce(w_avg_ask_price / w_avg_bid_price  -  1)

            del w_avg_bid_price
            del w_avg_ask_price
            torch.cuda.empty_cache()

            executed_px = torch.nan_to_num(executed_px) + torch.isnan(executed_px) * torch.nan_to_num(WAP)

            log_returns = torch.diff(torch.log(WAP))
            realized_vol = torch.sqrt(torch.sum(torch.pow(log_returns,2),dim=1))
            
            log_returns_windows = log_returns.unfold(1,kernel_size,1)
            realized_vol = reduce(torch.sqrt(torch.sum(torch.pow(log_returns_windows,2),dim=2))) #TODO

            log_returns = reduce(log_returns)

            mid_price = (bid_px1 + ask_px1) / 2
            mid_pricetoWAP = reduce(mid_price / WAP -1)

            spread =  reduce(ask_px1 / bid_px1 - 1)  # TODO: mejor formula para el spread?

            del log_returns_windows
            del mid_price
            del bid_px1
            del ask_px1
            del bid_qty1
            del ask_qty1

            gc.collect()
            torch.cuda.empty_cache()

            ################################ Inspired on Technical Analysis ################################

            # Inspired on https://www.investopedia.com/terms/o/onbalancevolume.asp
            OBV = reduce(((executed_qty[:,1:] * (torch.diff(executed_px) > 0)) - (executed_qty[:,1:] * (torch.diff(executed_px) < 0))) / stock_means)

            # Inspired on https://www.investopedia.com/terms/f/force-index.asp
            force_index = reduce(torch.diff(torch.log(executed_px)) * executed_qty[:,1:] * 1e5 / stock_means)

            # TODO: ??????????
            # executed_qty_windows = executed_qty.unfold(1,kernel_size,1)
            # moving_executed_qty = torch.mean(executed_qty_windows, axis=2)
            # effort = (executed_qty[:,1:] / log_returns) / moving_executed_qty
            # effort = torch.Tensor(torch.nan_to_num(effort))
            # effort = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)(effort.unsqueeze(0)).squeeze()


            # Inspired on https://www.investopedia.com/terms/m/mfi.asp
            
            money_flow = (executed_px * executed_qty)[:,1:]

            money_flow_pos = money_flow * (torch.diff(executed_px) > 0)
            money_flow_neg = money_flow * (torch.diff(executed_px) < 0)

            money_flow_pos_windows = money_flow_pos.unfold(1,kernel_size,1)
            money_flow_neg_windows = money_flow_neg.unfold(1,kernel_size,1)

            money_flow_pos_sums = torch.nansum(money_flow_pos_windows, axis=2)
            money_flow_neg_sums = torch.nansum(money_flow_neg_windows, axis=2)

            MFI = money_flow_pos_sums / (money_flow_pos_sums + money_flow_neg_sums)
            MFI = reduce(torch.nan_to_num(MFI,nan=0.5), 'MFI    ')

            del money_flow_pos
            del money_flow_neg
            del money_flow_pos_windows
            del money_flow_neg_windows
            del money_flow_pos_sums
            del money_flow_neg_sums
            torch.cuda.empty_cache()


            # Inspired on https://www.investopedia.com/terms/v/vwap.asp

            money_flows_windows = money_flow.unfold(1,kernel_size,1)
            executed_qty_windows = executed_qty[:,1:].unfold(1,kernel_size,1)

            VWAP = torch.sum(money_flows_windows, axis=2) / torch.sum(executed_qty_windows, axis=2)
            VWAP = torch.tensor(pd.DataFrame(VWAP.cpu().numpy()).ffill(axis=1).values, device=device)
            VWAP = torch.nan_to_num(VWAP) + torch.isnan(VWAP) * torch.nan_to_num(WAP[:,:VWAP.shape[1]])  # beginning of series
            VWAPtoWAP = reduce(VWAP / WAP[:,-VWAP.shape[1]:] - 1)

            del VWAP
            del money_flows_windows
            del executed_qty_windows
            torch.cuda.empty_cache()


            ####

            WAP_windows = WAP.unfold(1,kernel_size,1)

            moving_mean = torch.mean(WAP_windows, axis=2)
            moving_std = torch.std(WAP_windows, axis=2)
            moving_min = torch.min(WAP_windows, axis=2).values
            moving_max = torch.max(WAP_windows, axis=2).values

            del WAP_windows
            torch.cuda.empty_cache()


            # Inspired on https://www.investopedia.com/terms/u/.....

            bollinger_deviation = reduce(((moving_mean - WAP[:,-moving_mean.shape[1]:]) / (moving_std + epsilon)))  #TODO moving_std==0 case

            del moving_mean
            moving_std = reduce(moving_std)


            # Inspired on https://www.investopedia.com/terms/u/ulcerindex.asp
            R = reduce(torch.pow(100 * (WAP[:,-moving_max.shape[1]:] - moving_max) / moving_max, 2), 'R       ')

            # Inspired on https://www.investopedia.com/articles/trading/08/accumulation-distribution-line.asp
            CLV = torch.nan_to_num(((executed_px[:,-moving_min.shape[1]:] - moving_min) - (moving_max - executed_px[:,-moving_max.shape[1]:])) /       \
                            (moving_max - moving_min), nan=0., posinf=0, neginf=0)

            # Inspired on ......
            ATR = reduce(moving_max / moving_min - 1)

            # Inspired on ......
            middle_channel = (moving_max + moving_min) / 2
            donchian_deviation = reduce((middle_channel - WAP[:,-middle_channel.shape[1]:]) / (moving_max - moving_min + epsilon))  #TODO

            del middle_channel
            del moving_min
            del moving_max
            torch.cuda.empty_cache()


            # Inspired on https://www.investopedia.com/terms/c/chaikinoscillator.asp
            lexqty = executed_qty[:,-CLV.shape[1]:]
            execlv_windows = (lexqty*CLV).unfold(1,kernel_size,1)
            vol_windows = lexqty.unfold(1,kernel_size,1)
            CMF = torch.nan_to_num(torch.mean(execlv_windows, axis=2) / torch.mean(vol_windows, axis=2), nan=0., posinf=0, neginf=0)

            del executed_qty
            del lexqty
            del execlv_windows
            del vol_windows
            torch.cuda.empty_cache()

            CLV = reduce(CLV, 'CLV    ')
            CMF = reduce(CMF, 'CMF    ')

            executed_pxtoWAP = reduce(executed_px / WAP - 1)
            del executed_px

            WAP = reduce(WAP, 'WAP    ')

            #TODO: trades

            # # #"some commonly seen technical signals of the financial market are derived from trade data directly, such as high-low or total traded volume." segun organizadores

            # last_wavg_px = np.nan_to_num(last_wavg_px) + np.isnan(last_wavg_px) * np.nan_to_num(WAP)
            # # last_log_returns = np.diff(np.log(last_wavg_px), prepend=0)  # TODO: double-check prepend
            # # log_returns_dev2 = (log_returns + epsilon) / (last_log_returns + epsilon) - 1

            torch.cuda.empty_cache()

            nels = min(
                        10000000000,

                        moving_std.shape[1],
                        realized_vol.shape[1],

                        log_returns.shape[1],
                        spread.shape[1],

                        deepWAPtoWAP.shape[1],
                        executed_pxtoWAP.shape[1],
                        mid_pricetoWAP.shape[1],
                        VWAPtoWAP.shape[1],

                        vol_total.shape[1],
                        vol_unbalance1.shape[1],
                        vol_unbalance2.shape[1],
                        vol_unbalance3.shape[1],

                        OBV.shape[1],
                        force_index.shape[1],
                        MFI.shape[1],
                        bollinger_deviation.shape[1],
                        R.shape[1],
                        ATR.shape[1],
                        donchian_deviation.shape[1],
                        CLV.shape[1],
                        CMF.shape[1],

                        OFI1.shape[1],
                        OFI2.shape[1]
                        )

            processed.append(np.stack((

                # moving_std[:,-nels:],    # 0.31 <----------------------
                # realized_vol[:,-nels:],    # 0.31 <----------------------

                # log_returns[:,-nels:],    # 0.35 <----------------------
                # spread[:,-nels:],    # 0.42 <----------------------

                # deepWAPtoWAP[:,-nels:],    # 0.48
                # executed_pxtoWAP[:,-nels:],    # 0.48
                # mid_pricetoWAP[:,-nels:],    # 0.46
                # VWAPtoWAP[:,-nels:],    # 0.38 <----------------------

                vol_total[:,-nels:],   # 0.53
                vol_unbalance1[:,-nels:],
                vol_unbalance2[:,-nels:],
                vol_unbalance3[:,-nels:],

                # OBV[:,-nels:],   # 0.54
                # force_index[:,-nels:],   # 0.51
                # MFI[:,-nels:],   # 0.53
                # bollinger_deviation[:,-nels:],   # 0.53
                # R[:,-nels:],   # 0.38 <----------------------
                # ATR[:,-nels:],   # 0.33 <----------------------
                # donchian_deviation[:,-nels:],   # 0.53
                # CLV[:,-nels:],   # 0.53
                # CMF[:,-nels:],   # 0.53

                # OFI1[:,-nels:],   # 0.5
                # OFI2[:,-nels:]

            ), axis=1))

            gc.collect()

        series = np.vstack(processed)

        print('min:',list(np.round(np.min(series,axis=(0,2)),4)))
        print('max:',list(np.round(np.max(series,axis=(0,2)),4)))

        series = (series - np.expand_dims(np.mean(series, axis=(0,2)),1)) / np.expand_dims(np.std(series, axis=(0,2)),1)

        ###############################################################################

        print('computing stats...')

        l = series.shape[-1]

        all_means = np.mean(series,axis=2)
        hal_means = np.mean(series[:,:,-l//2:],axis=2)
        qua_means = np.mean(series[:,:,-l//4:],axis=2)

        all_std = np.std(series,axis=2)
        hal_std = np.std(series[:,:,-l//2:],axis=2)
        qua_std = np.std(series[:,:,-l//4:],axis=2)

        all_abmax = np.max(np.abs(series),axis=2)
        hal_abmax = np.max(np.abs(series[:,:,-l//2:]),axis=2)
        qua_abmax = np.max(np.abs(series[:,:,-l//4:]),axis=2)

        #kk = np.apply_along_axis(lambda x : np.sqrt(np.sum(x**2)), 1, series[:,2])
        # kk = (kk - np.mean(kk)) / (np.std(kk) + epsilon)
        #kk = np.expand_dims(kk,1)

        self.stats = np.hstack((all_means, hal_means, qua_means,
                                all_std, hal_std, qua_std,
                                all_abmax, hal_abmax, qua_abmax))

        scaler = StandardScaler().fit(self.stats[self.targets[:,0] % 5 != 0])
        self.stats = scaler.transform(self.stats)

        self.series = series

        # for i in range(NUM_SERIES):

        #     scaler = MinMaxScaler().fit(s[:,i].T)
        #     s[:,i] = scaler.transform(s[:,i].T).T


    def train_dataloader(self):

        return DataLoader(SeriesDataSet(self.series, self.stats, self.targets[self.targets[:,0] % 5 != 0]), batch_size=512, shuffle=True)

    def val_dataloader(self):

        return DataLoader(SeriesDataSet(self.series, self.stats, self.targets[self.targets[:,0] % 5 == 0]), batch_size=1024, shuffle=False)

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

                        # For spans with no trades, values of traded_qty and traded_count are set to 0
                        trades_columns = np.repeat(np.nan, 3*600).reshape(3,600).astype(np.float32)
                        trades_columns[-2:] = 0.

                        series.append(np.vstack((df_time.T.to_numpy(dtype=np.float32), trades_columns)))

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


