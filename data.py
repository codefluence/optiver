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
import torch.nn.functional as F

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import MinMaxScaler

from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.signal import argrelmax

np.set_printoptions(threshold=2000, linewidth=140, precision=6, edgeitems=20, suppress=1)
pd.set_option('display.max_rows', 600)



class OptiverDataModule(pl.LightningDataModule):

    def __init__(self, settings_path='./settings.json', device='cuda', batch_size=50000, scale=True,
                 kernel_size=5, stride=5, mks=30, CV_split=0, end_clip=30):

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



        series  = series[::2]
        targets = targets[::2]



        signals = np.empty((len(series),2), dtype=np.float32)
        signals[:] = np.nan



        print('computing stock stats...')

        stats = np.empty((len(series),2), dtype=np.float32)
        stats[:] = np.nan

        # sum of executed_qty for each data point
        stats[:,0] = np.sum(series[:,9], axis=1)

        texecqty_mean_all = np.empty((len(series)), dtype=np.float32)
        texecqty_mean_all[:] = np.nan

        # for each stock id, the mean of the sum of executed_qty is computed
        # this number tells abouts the usual volume of each stock
        for i in np.unique(targets[:,0]):
            idx = targets[:,0] == int(i)
            texecqty_mean_all[idx] = np.mean(stats[idx,0])

        stats[:,0] = stats[:,0] / texecqty_mean_all

        print('processing series...')

        processed = []

        num_batches = math.ceil(len(series) / batch_size)

        for bidx in tqdm(range(num_batches)):

            batch = []

            start = bidx*batch_size
            end   = start + min(batch_size, len(series) - start)

            def process(raw, toinclude):

                windows = raw.unfold(1,mks,1)

                rolling_mean = torch.mean(windows, dim=2)
                rolling_std  = torch.std(windows, dim=2)
                rolling_max  = torch.max(windows, dim=2)[0]
                rolling_min  = torch.min(windows, dim=2)[0]

                del windows

                results = []

                if 0 in toinclude: results.append(raw)
                if 1 in toinclude: results.append(rolling_mean)
                if 2 in toinclude: results.append(rolling_std)#TODO: nans?
                if 3 in toinclude: results.append(rolling_max)
                if 4 in toinclude: results.append(rolling_min)

                if 5 in toinclude:
                    rolling_deviation = (raw[:,-rolling_mean.shape[1]:] - rolling_mean) / rolling_std
                    results.append(torch_nan_to_num(rolling_deviation, nan=0, posinf=0, neginf=0))

                if 6 in toinclude:
                    rolling_minmaxspread = rolling_max / rolling_min -1
                    results.append(torch_nan_to_num(rolling_minmaxspread, nan=0, posinf=0, neginf=0))

                if 7 in toinclude:
                    h = rolling_deviation.shape[1] // 2
                    signals[start:end,0] = ((torch.sum((rolling_deviation[:,:h] > 5) | (rolling_deviation[:,:h] < -5),dim=1)>0)*1).cpu().numpy()
                    signals[start:end,1] = ((torch.sum((rolling_deviation[:,-h:] > 5) | (rolling_deviation[:,-h:] < -5),dim=1)>0)*1).cpu().numpy()

                #TODO: con una sola linea peta
                for i in range(len(results)):

                    # copy is necessary?
                    results[i] = F.avg_pool1d(results[i].unsqueeze(0), kernel_size=kernel_size,
                                              stride=stride).squeeze(0).cpu().numpy().copy()

                    assert((~np.isfinite(results[i])).sum() == 0)

                batch.extend(results)

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
            order_count = torch.tensor(series[start:end,10], device=device)

            texecqty_mean = torch.tensor(texecqty_mean_all[start:end], device=device).unsqueeze(1)

            # executed_qty_dist = torch.divide(executed_qty, texecqty_mean)
            # executed_qty_dist = torch_nan_to_num(executed_qty_dist, posinf=0, neginf=0)
            #process(executed_qty_dist)
            
            # order_count_dist = torch.divide(order_count,torch.sum(order_count,axis=1).unsqueeze(1))
            # order_count_dist = torch_nan_to_num(order_count_dist, posinf=0, neginf=0)
            #process(order_count_dist)

            #process(torch.divide(order_count, texecqty_mean))  #WORSE

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

            # OLI =   get_liquidityflow(bid_px1, bid_qty1 / texecqty_mean, True) + \
            #         get_liquidityflow(ask_px1, ask_qty1 / texecqty_mean, False) + \
            #         get_liquidityflow(bid_px2, bid_qty2 / texecqty_mean, True) + \
            #         get_liquidityflow(ask_px2, ask_qty2 / texecqty_mean, False)

            #process(OLI)

            t_bid_size = bid_qty1 + bid_qty2
            t_ask_size = ask_qty1 + ask_qty2

            # vol_total_diff = reduce(torch_diff(torch.log(t_bid_size + t_ask_size)))

            # vol_unbalance1 = reduce(torch_diff(t_ask_size / ( t_ask_size + t_bid_size + epsilon)))
            # vol_unbalance2 = reduce(torch_diff((ask_qty2 + bid_qty2 - ask_qty1 - bid_qty1) / (texecqty_mean + epsilon)))
            # vol_unbalance3 = reduce(torch_diff((ask_qty1 + bid_qty2 - ask_qty2 - bid_qty1) / (texecqty_mean + epsilon)))

            # # #"some commonly seen technical signals of the financial market are derived from trade data directly, such as high-low or total traded volume." segun organizadores
            # last_wavg_px = np.nan_to_num(last_wavg_px) + np.isnan(last_wavg_px) * np.nan_to_num(WAP)
            # # last_log_returns = np.diff(np.log(last_wavg_px), prepend=0)  # TODO: double-check prepend
            # # log_returns_dev2 = (log_returns + epsilon) / (last_log_returns + epsilon) - 1

            ##########################################################################################################

            WAP = (bid_px1*ask_qty1 + ask_px1*bid_qty1) / (bid_qty1 + ask_qty1)
            process(WAP,[0,2,5,6,7])####################
            ########################## BAJO CONTROL #############################
            # # Inspired on https://www.investopedia.com/terms/u/.....
            # bollinger_deviation = (WAP_rolling_mean - WAP[:,-WAP_rolling_mean.shape[1]:]) > (WAP_rolling_std + epsilon)*2
            # process(bollinger_deviation,start,end)
            # del moving_mean
            ########################## BAJO CONTROL #############################
            # Inspired on ......
            #ATR
            #process(WAP_rolling_max / WAP_rolling_min - 1)####################

            log_returns = torch_diff(torch.log(WAP))
            process(log_returns,[0,2,3,5])#################### 0y1, 3y6 redundantes?

            # print()
            # print('Volatility decreases in general:')
            # print('first 100 seconds:', round(torch.mean(torch.std(log_returns[:,:100], dim=1)*1e3).item(),3))
            # print('last 100 seconds:', round(torch.mean(torch.std(log_returns[:,-100:], dim=1)*1e3).item(),3))
            # print('drop in the last 30 seconds:', round(torch.mean(torch.std(log_returns[:,-30:], dim=1)*1e3).item(),3))

            executed_px = torch_nan_to_num(executed_px) + torch.isnan(executed_px) * torch_nan_to_num(WAP)
            #executed_px_returns = torch_diff(torch.log(executed_px))
            #process(executed_px_returns) #NO gran mejoria respecto log_returns, PROBAR en lugar del ratio con WAP?

            # past realized volatility
            stats[start:end,1] = torch.sqrt(torch.sum(torch.pow(log_returns,2),dim=1)).cpu().numpy()

            # moving_realized_vols = []

            # for mks in m_kernel_sizes:

            #     realized_vols_windows = log_returns.unfold(1,mks,1)

            #     moving_realized_vols.append(reduce(torch.sqrt(torch.sum(torch.pow(log_returns.unfold(1,mks,1),2),dim=2))))

            #     del realized_vols_windows

            w_avg_bid_price = (bid_px1*bid_qty1 + bid_px2*bid_qty2) / t_bid_size
            w_avg_ask_price = (ask_px1*ask_qty1 + ask_px2*ask_qty2) / t_ask_size

            deepWAP = (w_avg_bid_price * t_ask_size + w_avg_ask_price * t_bid_size) / (t_bid_size + t_ask_size)
            #deepWAP_returns = torch_diff(torch.log(deepWAP))
            #process(deepWAP_returns) #NO gran mejoria respecto log_returns, PROBAR en lugar del ratio con WAP?

            midprice = (bid_px1 + ask_px1) / 2
            #midprice_returns = torch_diff(torch.log(midprice))
            #process(midprice_returns) #NO gran mejoria respecto log_returns, PROBAR en lugar del ratio con WAP?

            #spread
            process(ask_px1 / bid_px1 - 1,[0])####################

            #price dev
            process(deepWAP / WAP - 1,[0])####################
            process(midprice / WAP - 1,[0])####################
            process(executed_px / WAP - 1,[0])####################

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
            # del deepWAP
            # del midprice


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
            
            money_flow = (executed_px * executed_qty)[:,1:]

            # money_flow_pos = money_flow * (torch_diff(executed_px) > 0)
            # money_flow_neg = money_flow * (torch_diff(executed_px) < 0)

            # money_flow_pos_windows = money_flow_pos.unfold(1,mks,1)
            # money_flow_neg_windows = money_flow_neg.unfold(1,mks,1)

            # money_flow_pos_sums = torch.nansum(money_flow_pos_windows, axis=2)
            # money_flow_neg_sums = torch.nansum(money_flow_neg_windows, axis=2)

            # MFI = money_flow_pos_sums / (money_flow_pos_sums + money_flow_neg_sums)
            # process(torch_nan_to_num(MFI,nan=0.5))

            # del money_flow_pos
            # del money_flow_neg
            # del money_flow_pos_windows
            # del money_flow_neg_windows
            # del money_flow_pos_sums
            # del money_flow_neg_sums
            # #del MFI


            # # Inspired on https://www.investopedia.com/terms/v/vwap.asp

            money_flows_windows = money_flow.unfold(1,mks,1)
            executed_qty_windows = executed_qty[:,1:].unfold(1,mks,1)

            VWAP = torch.sum(money_flows_windows, axis=2) / torch.sum(executed_qty_windows, axis=2)
            VWAP = torch.tensor(pd.DataFrame(VWAP.cpu().numpy()).ffill(axis=1).values, device=device)
            VWAP = torch_nan_to_num(VWAP) + torch.isnan(VWAP) * torch_nan_to_num(WAP[:,-VWAP.shape[1]:])  # beginning of series
            process(WAP,[0,2,5,6])###################

            VWAP_returns = torch_diff(torch.log(VWAP))
            process(VWAP_returns,[0,2,3,5])####################
            #process(VWAP / WAP[:,-VWAP.shape[1]:] - 1)####################

            del money_flow
            del money_flows_windows
            del executed_qty_windows
            #del WAP


            # ################################################################################################################################


            #TODO: lo mismo con VWAP
            #TODO: lo mismo con log_returns, deepWAP_returns y midprice_returns? lo mismo con otros precios??
            WAP_windows = WAP.unfold(1,mks,1)

            WAP_rolling_mean = torch.mean(WAP_windows, axis=2)
            WAP_rolling_std  = torch.std(WAP_windows, axis=2)
            WAP_rolling_max  = torch.max(WAP_windows, axis=2).values
            WAP_rolling_min  = torch.min(WAP_windows, axis=2).values

            process(WAP_rolling_std,[1,3,5])  #TODO: 3 y 5 son utiles?

            del WAP_windows

            # ####

            # # Inspired on https://www.investopedia.com/terms/u/ulcerindex.asp
            R = torch.pow(100 * (WAP[:,-WAP_rolling_max.shape[1]:] - WAP_rolling_max) / WAP_rolling_max, 2)
            process(R,[0])

            # # Inspired on https://www.investopedia.com/articles/trading/08/accumulation-distribution-line.asp
            # CLV = ((executed_px[:,-WAP_rolling_min.shape[1]:] - WAP_rolling_min) - (WAP_rolling_max - executed_px[:,-WAP_rolling_max.shape[1]:])) /   \
            #                  (WAP_rolling_max - WAP_rolling_min)
            # process(torch_nan_to_num(CLV, nan=0, posinf=0, neginf=0),[0])

            # # Inspired on ......
            middle_channel = (WAP_rolling_max + WAP_rolling_min) / 2
            donchian_deviation = (middle_channel - WAP[:,-middle_channel.shape[1]:]) / (WAP_rolling_max - WAP_rolling_min)  #TODO
            process(torch_nan_to_num(donchian_deviation, nan=0, posinf=0, neginf=0),[0])

            del middle_channel
            del WAP_rolling_min
            del WAP_rolling_max

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

            del executed_px
            del WAP #TODO: antes?


            s_lengths = [102 + end_clip//stride]#TODO

            for s in batch:
                s_lengths.append(s.shape[1])

            for i in range(len(batch)):
                batch[i] = batch[i][:,-min(s_lengths):-end_clip//stride]  #TODO

            processed.append(np.stack(batch, axis=1))

            gc.collect()
            torch.cuda.empty_cache()

        series = np.vstack(processed)
        gc.collect()

        print('min:',list(np.round(np.min(series,axis=(0,2)),5)))
        #print('med:',list(np.round(np.median(series,axis=(0,2)),5)))
        print('max:',list(np.round(np.max(series,axis=(0,2)),5)))

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

        #TODO
        # for i in range(series.shape[1]):

        #     scaler = MinMaxScaler().fit(series[:,i].T)
        #     series[:,i] = scaler.transform(series[:,i].T).T

        series_medians = series_means#TODO:np.median(series, axis=0)

        assert((~np.isfinite(series)).sum() == 0)
        assert((~np.isfinite(series_medians)).sum() == 0)

        print('series shape:',series.shape)

        self.series = series
        self.series_medians = series_medians

        ###############################################################################

        print('computing targets...')

        fut_rea_vol = targets[:,3]
        past_rea_vol = stats[:,1]
        rea_vol_delta = fut_rea_vol - past_rea_vol
        rea_vol_increase = fut_rea_vol / past_rea_vol - 1

        targets = np.hstack((   targets,
                                np.expand_dims(past_rea_vol,1),
                                np.expand_dims(rea_vol_delta,1),
                                np.expand_dims(rea_vol_increase,1)
                            ))

        assert((~np.isfinite(targets)).sum() == 0)

        self.targets = targets

        ###############################################################################

        print('computing series stats...')

        stats_h = []

        # r = series.shape[2]
        # ds = [1, 2, 4]
        # for d in ds:

        #     s = series[:,:,-r//d:]

        #     stats_h.append(np.hstack((

        #         np.mean(s,axis=2),
        #         np.std(s,axis=2),
        #         np.max(s,axis=2),
        #         np.min(s,axis=2)
                    
        #      )))

        #     corrs = np.corrcoef(np.hstack((stats_h[-1], np.expand_dims(targets[:,-1],1))).T)[:-1,-1]
        #     print('corrs for last {} ticks:'.format(r//d), np.round(corrs,2))

        # # corrs = np.corrcoef(np.hstack((stats_h[2]-stats_h[1], np.expand_dims(targets[:,-1],1))).T)[:-1,-1]
        # # print('corrs for {} t - {} t:'.format(r//ds[2], r//ds[1]), np.round(corrs,2))
        # corrs = np.corrcoef(np.hstack((stats_h[2]-stats_h[0], np.expand_dims(targets[:,-1],1))).T)[:-1,-1]
        # print('corrs for {} t - {} t:'.format(r//ds[2], r//ds[0]), np.round(corrs,2))

        stats_h.append(stats)
        stats_h.append(signals)#######################
        stats = np.hstack(stats_h)

        corrs = np.corrcoef(np.hstack((stats, np.expand_dims(targets[:,-1],1))).T)[:-1,-1]
        print('abs corrs sorted:')
        print(np.round(np.sort(np.abs(corrs))[::-1],2))

        THRESHOLD = 0.
        stats = stats[:,np.abs(corrs)>THRESHOLD]  #TODO
        print('stats shape',stats.shape)

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

        print('coeffs of selected stats:')
        print(np.round(np.corrcoef(np.hstack((stats, np.expand_dims(targets[:,-1],1))).T)[:-1,-1],2))

        assert((~np.isfinite(stats)).sum() == 0)

        self.stats = stats

    def train_dataloader(self):

        #TODO: sorting por volatilidad
        idx = self.targets[:,0] % 5 != self.CV_split
        return DataLoader(SeriesDataSet(self.series[idx], self.stats[idx], self.targets[idx]), batch_size=4096, shuffle=True)

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

                # def get_extrema(array):
                #     #TODO: assert((~ torch.isfinite(array)).sum() == 0)
                #     h = array.shape[1]

                #     batch2.append(F.max_pool1d(array[:,:h//2].unsqueeze(0), array[:,:h//2].shape[1], 1).squeeze(0).cpu().numpy().copy())
                #     batch2.append(F.max_pool1d(array[:,h//2:-h//4].unsqueeze(0), array[:,h//2:-h//4].shape[1], 1).squeeze(0).cpu().numpy().copy())
                #     batch2.append(F.max_pool1d(array[:,-h//4:].unsqueeze(0), array[:,-h//4:].shape[1], 1).squeeze(0).cpu().numpy().copy())

                #     batch2.append(F.max_pool1d(-1*array[:,:h//2].unsqueeze(0), array[:,:h//2].shape[1], 1).squeeze(0).cpu().numpy().copy())
                #     batch2.append(F.max_pool1d(-1*array[:,h//2:-h//4].unsqueeze(0), array[:,h//2:-h//4].shape[1], 1).squeeze(0).cpu().numpy().copy())
                #     batch2.append(F.max_pool1d(-1*array[:,-h//4:].unsqueeze(0), array[:,-h//4:].shape[1], 1).squeeze(0).cpu().numpy().copy())

                # empty = torch.zeros((raw.shape[0],raw.shape[1]-1), device='cuda')
                # empty[:] = np.nan
                # reversedextended = torch.flip(torch.hstack((raw,empty)),dims=(1,))
                # windows = reversedextended.unfold(1,raw.shape[1],1).cpu().numpy()
                # torch.cuda.empty_cache()

                # md = np.nanmean(windows, axis=2)[:,::-1].copy()
                # results.append(torch.tensor(md, device='cuda' ))
                # ss = np.nanstd(windows, axis=2)[:,::-1].copy()
                # results.append(torch.tensor(ss, device='cuda' ))


                    # #TODO: post-pro
                    # def caca(a_series):

                    #     peaks = argrelmax(a_series, order=2, axis=0)[0]

                    #     if not 0 in peaks:
                    #         peaks = np.append(peaks, [0])

                    #     if not (len(a_series)-1) in peaks:
                    #         peaks = np.append(peaks, [len(a_series)-1])

                    #     try:
                    #         f = interp1d(peaks, a_series[peaks], kind='cubic')
                    #     except:
                    #         return a_series

                    #     return f(range(len(a_series)))
                    
                    # results[i] = np.apply_along_axis(caca, 1, results[i])



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