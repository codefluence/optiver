import os
import gc
import json
import math
import pickle
import platform
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

import torch
import torch.nn as nn

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import MinMaxScaler

np.set_printoptions(threshold=2000, linewidth=140, precision=6, edgeitems=20, suppress=1)
pd.set_option('display.max_rows', 600)



class OptiverDataModule(pl.LightningDataModule):

    def __init__(self, settings_path='./settings.json', device='cuda', batch_size=50000, scale=True,
                 horizon=-243-8, kernel_size=3, stride=2, m_kernel_sizes=[30], CV_split=0, correction=True):

        super(OptiverDataModule, self).__init__()

        self.CV_split = CV_split

        epsilon = 1e-6

        with open(settings_path) as f:
            
            settings = json.load(f)

        if settings['ENV'] == 'train':

            if not os.path.exists(settings['DATA_DIR'] +'optiver_series.npz'):

                print('creating tensors file...')
                series, targets = get_time_series(settings)

                np.savez_compressed(settings['DATA_DIR'] + 'optiver_series.npz', series=series, targets=targets)

            print('reading tensors file...')
            tensors = np.load(settings['DATA_DIR'] + 'optiver_series.npz')

            targets = tensors['targets']
            series  = tensors['series']  # shape(428932, 11, 600)
        else:

            print('creating tensors...')
            series, targets = get_time_series(settings)

        #TODO: remove
        ##############################################
        series[:,-2:] = np.nan_to_num(series[:,-2:])
        series[:,8][series[:,8]==0] = 1
        ##############################################

        assert((~ np.isfinite(series[:,:8])).sum() == 0)
        assert((~ np.isfinite(series[:,-2:])).sum() == 0)



        print('computing stock stats...')

        stats = np.empty((len(series),3), dtype=np.float32)
        stats[:] = np.nan

        # sum of executed_qty for each data point
        stats[:,0] = np.sum(series[:,9], axis=1)  

        # for each stock id, the mean of the sum of executed_qty is computed
        # this number tells abouts the usual volume of each stock
        for i in np.unique(targets[:,0]):
            idx = targets[:,0] == int(i)
            stats[idx,1] = np.mean(stats[idx,0])



        print('processing series...')

        def reduce(array):

            results = []
            results.append(array)

            for mks in m_kernel_sizes:

                windows = torch.cumsum(array, dim=1).unfold(1,mks,1)

                results.append(torch.mean(windows, axis=2))
                results.append(torch.std(windows, axis=2))
                results.append(torch.max(windows, axis=2))
                results.append(torch.max(windows, axis=2) - torch.min(windows, axis=2))

                del windows

            #TODO: con una sola linea peta
            for kk in range(len(results)):

                # copy is necessary?
                results[kk] = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)(results[kk].unsqueeze(0)).squeeze(0).cpu().numpy().copy()[:,horizon:]
                assert((~ np.isfinite(results[kk])).sum() == 0)

            return results

        # Inspired on ...
        def get_liquidityflow(prices, qties, isbuy):

            #torch.cuda.empty_cache()

            price_diff = torch_diff(prices)
            vol_diff   = torch_diff(qties)

            if isbuy:

                return  (price_diff == 0) * vol_diff + \
                        (price_diff > 0)  * qties[:,1:] - \
                        (price_diff < 0)  * torch.roll(qties, 1, 1)[:,1:]

            else:

                return  (price_diff == 0) * vol_diff - \
                        (price_diff > 0)  * torch.roll(qties, 1, 1)[:,1:] + \
                        (price_diff < 0)  * qties[:,1:]

        processed = []

        num_batches = math.ceil(len(series) / batch_size)

        for bidx in tqdm(range(num_batches)):

            start = bidx*batch_size
            end   = start + min(batch_size, len(series) - start)

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
            #order_count = torch.tensor(series[start:end,10], device=device)

            # executed_qty_dist =  reduce(np.divide(executed_qty, sumexecs))
            # order_count_dist = reduce(np.divide(order_count, np.expand_dims(np.sum(order_count,axis=1),axis=1)))

            texecqty_mean = torch.tensor(stats[start:end,1], device=device).unsqueeze(1)

            OLI1 = get_liquidityflow(bid_px1, bid_qty1 / texecqty_mean, True) + get_liquidityflow(ask_px1, ask_qty1 / texecqty_mean, False)
            OLI2 = get_liquidityflow(bid_px2, bid_qty2 / texecqty_mean, True) + get_liquidityflow(ask_px2, ask_qty2 / texecqty_mean, False)

            moving_OLIs = []

            for mks in m_kernel_sizes:

                OLI_windows = torch.cumsum(OLI1+OLI2, dim=1).unfold(1,mks,1)

                moving_OLIs.append(reduce(torch.mean(OLI_windows, axis=2)))

                del OLI_windows

            del OLI1
            del OLI2

            t_bid_size = bid_qty1 + bid_qty2
            t_ask_size = ask_qty1 + ask_qty2

            vol_total_diff = reduce(torch_diff(torch.log(t_bid_size + t_ask_size)))

            vol_unbalance1 = reduce(torch_diff(t_ask_size / ( t_ask_size + t_bid_size + epsilon)))
            vol_unbalance2 = reduce(torch_diff((ask_qty2 + bid_qty2 - ask_qty1 - bid_qty1) / (texecqty_mean + epsilon)))
            vol_unbalance3 = reduce(torch_diff((ask_qty1 + bid_qty2 - ask_qty2 - bid_qty1) / (texecqty_mean + epsilon)))

            WAP = (bid_px1*ask_qty1 + ask_px1*bid_qty1) / (bid_qty1 + ask_qty1)
            log_returns = torch_diff(torch.log(WAP))

            # print()
            # print('Volatility decreases in general:')
            # print('first 100 seconds:', round(torch.mean(torch.std(log_returns[:,:100], dim=1)*1e3).item(),3))
            # print('last 100 seconds:', round(torch.mean(torch.std(log_returns[:,-100:], dim=1)*1e3).item(),3))
            # print('drop in the last 30 seconds:', round(torch.mean(torch.std(log_returns[:,-30:], dim=1)*1e3).item(),3))

            executed_px = torch_nan_to_num(executed_px) + torch.isnan(executed_px) * torch_nan_to_num(WAP)
            executed_px_returns = reduce(torch_diff(torch.log(executed_px)))

            # past realized volatility
            stats[start:end,2] = torch.sqrt(torch.sum(torch.pow(log_returns,2),dim=1)).cpu().numpy()

            moving_realized_vols = []

            for mks in m_kernel_sizes:

                realized_vols_windows = log_returns.unfold(1,mks,1)

                moving_realized_vols.append(reduce(torch.sqrt(torch.sum(torch.pow(log_returns.unfold(1,m_kernel_sizes[1],1),2),dim=2))))

                del realized_vols_windows

            w_avg_bid_price = (bid_px1*bid_qty1 + bid_px2*bid_qty2) / t_bid_size
            w_avg_ask_price = (ask_px1*ask_qty1 + ask_px2*ask_qty2) / t_ask_size

            deepWAP = (w_avg_bid_price * t_ask_size + w_avg_ask_price * t_bid_size) / (t_bid_size + t_ask_size)
            deepWAP_returns = reduce(torch_diff(torch.log(deepWAP)))

            mid_price_returns = reduce(torch_diff(torch.log((bid_px1 + ask_px1) / 2)))

            spread = reduce(ask_px1 / bid_px1 - 1)
            deep_spread = reduce(w_avg_ask_price / w_avg_bid_price - 1)

            del bid_px1
            del ask_px1
            del bid_qty1
            del ask_qty1

            del bid_px2
            del ask_px2
            del bid_qty2
            del ask_qty2
 
            del t_bid_size
            del t_ask_size
            del w_avg_bid_price
            del w_avg_ask_price
            del deepWAP


            ################################ Inspired on Technical Analysis ################################

            # # Inspired on https://www.investopedia.com/terms/o/onbalancevolume.asp
            # OBV = reduce(((executed_qty[:,1:] * (torch_diff(executed_px) > 0)) - (executed_qty[:,1:] * (torch_diff(executed_px) < 0))) / texecqty_mean)

            # # Inspired on https://www.investopedia.com/terms/f/force-index.asp
            # # force_index = reduce(torch.diff(torch.log(executed_px)) * executed_qty[:,1:] * 1e5 / texecqty_mean)

            # # TODO: ??????????
            # # executed_qty_windows = executed_qty.unfold(1,m_kernel_sizes[1],1)
            # # moving_executed_qty = torch.mean(executed_qty_windows, axis=2)
            # # effort = (executed_qty[:,1:] / log_returns) / moving_executed_qty
            # # effort = torch.Tensor(torch.nan_to_num(effort))
            # # effort = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)(effort.unsqueeze(0)).squeeze()


            # # Inspired on https://www.investopedia.com/terms/m/mfi.asp
            
            # money_flow = (executed_px * executed_qty)[:,1:]

            # money_flow_pos = money_flow * (torch_diff(executed_px) > 0)
            # money_flow_neg = money_flow * (torch_diff(executed_px) < 0)

            # money_flow_pos_windows = money_flow_pos.unfold(1,m_kernel_sizes[1],1)
            # money_flow_neg_windows = money_flow_neg.unfold(1,m_kernel_sizes[1],1)

            # money_flow_pos_sums = torch.nansum(money_flow_pos_windows, axis=2)
            # money_flow_neg_sums = torch.nansum(money_flow_neg_windows, axis=2)

            # MFI = money_flow_pos_sums / (money_flow_pos_sums + money_flow_neg_sums)
            # MFI = reduce(torch_nan_to_num(MFI,nan=0.5))

            # del money_flow_pos
            # del money_flow_neg
            # del money_flow_pos_windows
            # del money_flow_neg_windows
            # del money_flow_pos_sums
            # del money_flow_neg_sums


            # # Inspired on https://www.investopedia.com/terms/v/vwap.asp

            # money_flows_windows = money_flow.unfold(1,m_kernel_sizes[1],1)
            # executed_qty_windows = executed_qty[:,1:].unfold(1,m_kernel_sizes[1],1)

            # VWAP = torch.sum(money_flows_windows, axis=2) / torch.sum(executed_qty_windows, axis=2)
            # VWAP = torch.tensor(pd.DataFrame(VWAP.cpu().numpy()).ffill(axis=1).values, device=device)
            # VWAP = torch_nan_to_num(VWAP) + torch.isnan(VWAP) * torch_nan_to_num(WAP[:,:VWAP.shape[1]])  # beginning of series
            # VWAP_returns = reduce(torch_diff(torch.log(VWAP)))

            # del money_flows_windows
            # del executed_qty_windows


            # ####

            moving_stds  = []
            moving_means = []
            moving_mins  = []
            moving_maxs  = []

            #TODO: lo mismo con deepWAP_returns y midprice_returns?
            for mks in m_kernel_sizes:

                WAP_windows = log_returns.unfold(1,mks,1)

                std = torch.std(WAP_windows, axis=2)

                if correction:

                    std = std - torch.median(std, dim=0)[0]

                moving_stds.append(reduce(std))

                moving_means.append(reduce(torch.mean(WAP_windows, axis=2)))
                moving_mins.append(reduce(torch.min(WAP_windows, axis=2).values))
                moving_maxs.append(reduce(torch.max(WAP_windows, axis=2).values))

                del WAP_windows

            #moving_stds 2nd level
            # moving_std_windows = moving_stds[2].unfold(1,60,1)

            # moving_std_mean = reduce(torch.mean(moving_std_windows, axis=2))
            # moving_std_std = reduce(torch.std(moving_std_windows, axis=2))

            # del moving_std_windows



            # # Inspired on https://www.investopedia.com/terms/u/.....

            # bollinger_deviation = reduce(((moving_mean - WAP[:,-moving_mean.shape[1]:]) / (moving_stds[1] + epsilon)))  #TODO moving_stds[1]==0 case

            # del moving_mean

            # moving_stds_diff = [None] * 3

            # moving_stds_diff[0] = torch_diff(moving_stds[0])
            # moving_stds_diff[1] = torch_diff(moving_stds[1])
            # moving_stds_diff[2] = torch_diff(moving_stds[2])

            # #moving_stds 2nd level
            # moving_std_diff_windows = moving_stds_diff[2].unfold(1,60,1)

            # moving_std_diff_mean = reduce(torch.mean(moving_std_diff_windows, axis=2))
            # moving_std_diff_std = reduce(torch.std(moving_std_diff_windows, axis=2))

            # del moving_std_diff_windows


            # moving_stds[0] = reduce(moving_stds[0])
            # moving_stds[1] = reduce(moving_stds[1])
            # moving_stds[2] = reduce(moving_stds[2])

            # moving_stds_diff[0] = reduce(moving_stds_diff[0])
            # moving_stds_diff[1] = reduce(moving_stds_diff[1])
            # moving_stds_diff[2] = reduce(moving_stds_diff[2])


            # ####

            # moving_std_VWAPs = []

            # for mks in m_kernel_sizes:

            #     VWAP_windows = VWAP.unfold(1,mks,1)

            #     moving_std_VWAPs.append(reduce(torch.std(VWAP_windows, axis=2)))

            #     del VWAP_windows

            # del VWAP

            # ####

            # # Inspired on https://www.investopedia.com/terms/u/ulcerindex.asp
            # R = reduce(torch.pow(100 * (WAP[:,-moving_max.shape[1]:] - moving_max) / moving_max, 2))

            # # Inspired on https://www.investopedia.com/articles/trading/08/accumulation-distribution-line.asp
            # CLV = torch_nan_to_num(((executed_px[:,-moving_min.shape[1]:] - moving_min) - (moving_max - executed_px[:,-moving_max.shape[1]:])) /       \
            #                 (moving_max - moving_min), nan=0., posinf=0, neginf=0)

            # # Inspired on ......
            # ATR = reduce(moving_max / moving_min - 1)

            # # Inspired on ......
            # middle_channel = (moving_max + moving_min) / 2
            # donchian_deviation = reduce((middle_channel - WAP[:,-middle_channel.shape[1]:]) / (moving_max - moving_min + epsilon))  #TODO

            # del middle_channel
            # del moving_min
            # del moving_max


            # # Inspired on https://www.investopedia.com/terms/c/chaikinoscillator.asp
            # lexqty = executed_qty[:,-CLV.shape[1]:]
            # execlv_windows = (lexqty*CLV).unfold(1,m_kernel_sizes[1],1)
            # vol_windows = lexqty.unfold(1,m_kernel_sizes[1],1)
            # CMF = torch_nan_to_num(torch.mean(execlv_windows, axis=2) / torch.mean(vol_windows, axis=2), nan=0., posinf=0, neginf=0)

            # del executed_qty
            # del lexqty
            # del execlv_windows
            # del vol_windows

            # CLV = reduce(CLV)
            # CMF = reduce(CMF)

            log_returns = reduce(log_returns)

            del executed_px
            del WAP#TODO: antes?

            # # #"some commonly seen technical signals of the financial market are derived from trade data directly, such as high-low or total traded volume." segun organizadores

            # last_wavg_px = np.nan_to_num(last_wavg_px) + np.isnan(last_wavg_px) * np.nan_to_num(WAP)
            # # last_log_returns = np.diff(np.log(last_wavg_px), prepend=0)  # TODO: double-check prepend
            # # log_returns_dev2 = (log_returns + epsilon) / (last_log_returns + epsilon) - 1


            processed.append(np.stack((

                # log_returns[:,horizon:],

                #np.cumsum(np.diff(moving_stds[2], n=60, axis=1), axis=1),#[:,horizon:],

                # moving_stds[0][:,horizon:],
                # moving_stds[1][:,horizon:],
                # moving_stds[2][:,horizon:],

                ######

                # moving_means[0][:,horizon:],

                # moving_realized_vols[0][:,horizon:],

                # moving_std_mean[:,horizon:],
                # moving_std_std[:,horizon:],
                # moving_stds_diff[0][:,horizon:],
                # moving_stds_diff[1][:,horizon:],
                # moving_stds_diff[2][:,horizon:],
                # moving_std_diff_mean[:,horizon:],
                # moving_std_diff_std[:,horizon:],

                # # Volume / liquidity (0.45)
                # moving_OLIs[0][:,horizon:],
                # vol_total_diff[:,-nels:],
                # vol_unbalance1[:,-nels:],
                # vol_unbalance2[:,-nels:],
                # vol_unbalance3[:,-nels:],

                # # Returns (0.26)
                # log_returns[:,-nels:],    # 0.286 <----------------------
                # deepWAP_returns[:,-nels:],    
                # executed_px_returns[:,-nels:],    # 0.48
                # mid_price_returns[:,-nels:],    # 0.46
                # VWAP_returns[:,horizon:],    # 0.38 <----------------------

                # # Volatility (0.27)
                # moving_stds[0][:,-nels:] - moving_stds[1][:,-nels:],
                # moving_stds[1][:,-nels:] - moving_stds[2][:,-nels:],
                # moving_std_VWAPs[0][:,-nels:] - moving_std_VWAPs[1][:,-nels:],
                # moving_std_VWAPs[1][:,-nels:] - moving_std_VWAPs[2][:,-nels:],
                # moving_std_VWAPs[0][:,-nels:],
                # moving_std_VWAPs[1][:,-nels:],
                # moving_std_VWAPs[2][:,-nels:],
                # realized_vol[:,-nels:],    # 0.31 <----------------------

                # # Spreads (0.26)
                # # spread[:,-nels:],    # 0.4 <----------------------
                # ATR[:,horizon:],   # 0.29 <----------------------
                # # R[:,-nels:],   # 0.38 <----------------------

                # OBV[:,horizon:],   # 0.54
                # #force_index[:,-nels:],   # 0.51
                # MFI[:,horizon:],   # 0.53
                # bollinger_deviation[:,horizon:],   # 0.53
                # donchian_deviation[:,horizon:],   # 0.53
                # CLV[:,horizon:],   # 0.53
                # CMF[:,horizon:],   # 0.53

            ), axis=1))

            gc.collect()

        series = np.vstack(processed)
        gc.collect()
        torch.cuda.empty_cache()

        print('min:',list(np.round(np.min(series,axis=(0,2)),4)))
        print('med:',list(np.round(np.median(series,axis=(0,2)),4)))
        print('max:',list(np.round(np.max(series,axis=(0,2)),4)))

        if scale and len(series) > 1:

            if settings['ENV'] == 'train':

                series_means = np.expand_dims(np.mean(series, axis=(0,2)),1)
                series_stds = np.expand_dims(np.std(series, axis=(0,2)),1)

                np.save(settings['PREPROCESS_DIR'] +'series_means', series_means)
                np.save(settings['PREPROCESS_DIR'] +'series_stds', series_stds)
            else:

                series_means = np.load(settings['PREPROCESS_DIR'] +'series_means.npy')
                series_stds = np.load(settings['PREPROCESS_DIR'] +'series_stds.npy')

            series = (series - series_means) / series_stds

        self.series = series

        ###############################################################################

        print('computing targets...')

        fut_rea_vol = targets[:,3]
        past_rea_vol = stats[:,2]
        rea_vol_delta = fut_rea_vol - past_rea_vol
        rea_vol_increase = fut_rea_vol / past_rea_vol - 1

        targets = np.hstack((   targets,
                                np.expand_dims(past_rea_vol,1),
                                np.expand_dims(rea_vol_delta,1),
                                np.expand_dims(rea_vol_increase,1)
                            ))

        print('computing series stats...')

        stats_h = []

        #l = series.shape[-1]
        h = [horizon, horizon//2, horizon//4]
        stat_funcs = [np.mean, np.std, np.max, np.min] # diff, np.mean(series_diff[:,:,h:, np.std(series_diff[

        #for s in series, np.diff(series, axis=2, prepend=np.array(1., dtype=np.float32)):
        s = series
        for f in stat_funcs:

            f0 = f(s[:,:,h[0]:],axis=2)
            f1 = f(s[:,:,h[1]:],axis=2)
            f2 = f(s[:,:,h[2]:],axis=2)

            stats_h.append(np.hstack(( f0, f1, f2, f1 - f0, f2 - f0 )))

            corrs = np.corrcoef(np.hstack((stats_h[-1], np.expand_dims(targets[:,-1],1))).T)[:-1,-1]

            print('stat_func:',str(f).split(' ')[1])
            print('corrs:',np.round(corrs,2))
            print()

        stats_h.append(stats)
        stats = np.hstack(stats_h)

        corrs = np.corrcoef(np.hstack((stats, np.expand_dims(targets[:,-1],1))).T)[:-1,-1]
        print(np.round(np.sort(np.abs(corrs)),2))

        THRESHOLD = 0.0
        stats = stats[:,np.abs(corrs)>THRESHOLD]
        print('final shape',stats.shape)

        if scale and len(stats) > 1:

            if settings['ENV'] == 'train':

                #TODO: usar solo los datos del train set
                stats_means = np.mean(stats, axis=0)
                stats_stds = np.std(stats, axis=0)

                np.save(settings['PREPROCESS_DIR'] +'stats_means', stats_means)
                np.save(settings['PREPROCESS_DIR'] +'stats_stds', stats_stds)
            else:

                stats_means = np.load(settings['PREPROCESS_DIR'] +'stats_means.npy')
                stats_stds = np.load(settings['PREPROCESS_DIR'] +'stats_stds.npy')

            stats = (stats - stats_means) / stats_stds

        self.stats = stats

        # print('coeffs de las columnas seleccionadas')
        # print(np.round(np.abs(np.corrcoef(np.hstack((stats, np.expand_dims(targets[:,-1],1))).T)[:-1,-1]),2))
        
        self.targets = targets

        assert((~np.isfinite(self.series)).sum() == 0)
        assert((~np.isfinite(self.stats)).sum() == 0)

    def train_dataloader(self):

        #TODO: sorting por volatilidad
        idx = self.targets[:,0] % 5 == self.CV_split
        return DataLoader(SeriesDataSet(self.series[idx], self.stats[idx], self.targets[idx]), batch_size=512, shuffle=True)

    def val_dataloader(self):

        idx = self.targets[:,0] % 5 == self.CV_split
        return DataLoader(SeriesDataSet(self.series[idx], self.stats[idx], self.targets[idx]), batch_size=44039, shuffle=False)



def torch_diff(tensor, hack=False):

    if hack:
        return torch.tensor(np.diff(tensor.cpu().numpy()))
    else:
        return torch.diff(tensor)

def torch_nan_to_num(tensor, nan=0., posinf=0., neginf=0., hack=False):

    if hack:
        return torch.tensor(np.nan_to_num(tensor.cpu().numpy(), nan=nan, posinf=posinf, neginf=neginf))
    else:
        return torch.nan_to_num(tensor, nan=nan, posinf=posinf, neginf=neginf)



def get_time_series(settings):

    series = []
    vols = []

    targets = pd.read_csv(settings['DATA_DIR'] + settings['ENV'] + '.csv')

    for folder in 'book_'+settings['ENV']+'.parquet', 'trade_'+settings['ENV']+'.parquet': 

        file_paths = []
        path_root = settings['DATA_DIR'] + folder + '/'

        for path, _, files in os.walk(path_root):
            for name in files:
                file_paths.append(os.path.join(path, name))

        for file_path in tqdm(file_paths):

            df = pd.read_parquet(file_path, engine='pyarrow')
            slash = '\\' if platform.system() == 'Windows' else '/'
            stock_id = int(file_path.split(slash)[-2].split('=')[-1])

            for time_id in np.unique(df.time_id):

                df_time = df[df.time_id == time_id].reset_index(drop=True)
                with_changes_len = len(df_time)

                # In kaggle public leaderboard, some books don't start with seconds_in_bucket=0
                # if 'book' in file_path:
                #     assert df_time.seconds_in_bucket[0] == 0

                df_time = df_time.reindex(list(range(600)))

                missing = set(range(600)) - set(df_time.seconds_in_bucket)
                df_time.loc[with_changes_len:,'seconds_in_bucket'] = list(missing)

                df_time = df_time.sort_values(by='seconds_in_bucket').reset_index(drop=True)

                if 'book' in file_path:

                    df_time = df_time.iloc[:,2:].ffill(axis=0)

                    # In kaggle public leaderboard, some books don't start with seconds_in_bucket=0
                    #TODO: workaround for https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/251775
                    df_time.bfill(axis=0, inplace=True)

                    # For spans with no trades, values of traded_qty and traded_count are set to 0
                    trades_columns = np.repeat(np.nan, 3*600).reshape(3,600).astype(np.float32)
                    trades_columns[-2:] = 0.

                    series.append(np.vstack((df_time.T.to_numpy(dtype=np.float32), trades_columns)))

                    if 'target' in targets.columns:
                        entry = targets.loc[(targets.stock_id==stock_id) & (targets.time_id==time_id), 'target']
                    else:
                        entrey = []

                    vols.append(np.array((  stock_id, time_id, len(vols), 
                                            entry.item() if len(entry)==1 else np.nan), dtype=np.float32))

                elif 'trade' in file_path:

                    if isinstance(vols, list):
                        series = np.stack(series, axis=0)
                        vols = np.stack(vols, axis=0)

                    # Avg trade prices are only forward-filled, nan values will be replaced with WAP later
                    df_time = df_time.iloc[:,2:].fillna({'size':0, 'order_count':0})
                    df_time.ffill(axis=0, inplace=True)

                    tensor_index = vols[(vols[:,0]==stock_id) & (vols[:,1]==time_id), 2].item()
                    series[int(tensor_index),-3:] = df_time.T.to_numpy(dtype=np.float32)

    return series, vols



class SeriesDataSet(Dataset):
    
    def __init__(self, series, stats, targets):

        super(SeriesDataSet, self).__init__()

        self.series = series
        self.stats = stats
        self.targets = targets

    def __len__(self):

        return len(self.series)

    def __getitem__(self, idx):

        return self.series[idx], self.stats[idx], self.targets[idx]


if __name__ == '__main__':

        # sliding_window_view not available for numpy < 1.20
        # moving_realized_volatility = np.apply_along_axis(lambda x: np.sqrt(np.sum(x**2)), arr=windows, axis=2)

    data = OptiverDataModule()

    # truth = pd.read_csv(DATA_DIR+'train.csv')

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


        #TODO: correlations

        # for i in range(NUM_SERIES):

        #     scaler = MinMaxScaler().fit(s[:,i].T)
        #     s[:,i] = scaler.transform(s[:,i].T).T