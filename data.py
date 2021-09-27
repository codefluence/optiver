import os
import gc
import json
import math
import pickle
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

from utils import get_time_series

np.set_printoptions(threshold=2000, linewidth=140, precision=5, edgeitems=20, suppress=1)
pd.set_option('display.max_rows', 600)



class OptiverDataModule(pl.LightningDataModule):

    def __init__(self, settings_path='./settings.json', device='cuda', batch_size=50000, scale=True,
                 kernel_size=3, stride=2, mks=30, CV_split=0, end_clip=0, fixed_length=300):

        super(OptiverDataModule, self).__init__()

        self.CV_split = CV_split

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

        assert((~np.isfinite(series[:,:8])).sum() == 0)
        assert((~np.isfinite(series[:,-2:])).sum() == 0)



        # Reduce the number of series for the experiments
        sirtride=2
        series  = series[::sirtride]
        targets = targets[::sirtride]



        print('computing stock stats...')

        stock_id = targets[:,0]

        # sum of executed_qty for each data point
        execqty_sum = np.sum(series[:,9], axis=1)
        execcount_sum = np.sum(series[:,10], axis=1)

        stock_execqty_mean = np.repeat(np.nan, len(series)).astype(np.float32)
        stock_execcount_mean = np.repeat(np.nan, len(series)).astype(np.float32)

        # for each stock id, the mean of the sum of executed_qty and executed_count is computed
        # these numbers tells about the usual volume /count of each stock
        for i in np.unique(stock_id):
            idx = stock_id == int(i)
            stock_execqty_mean[idx] = np.mean(execqty_sum[idx])
            stock_execcount_mean[idx] = np.mean(execcount_sum[idx])

        execqty_sum = execqty_sum / stock_execqty_mean
        execcount_sum = execcount_sum / stock_execcount_mean


        print('processing series and maps...')

        processed_series = []
        processed_maps = []
        #signals = []
        past_real_vol = np.repeat(np.nan, len(series)).astype(np.float32)

        num_batches = math.ceil(len(series) / batch_size)

        for bidx in tqdm(range(num_batches)):

            batch_series = []
            batch_maps = []
            #batch_signals = []

            start = bidx*batch_size
            end   = start + min(batch_size, len(series) - start)

            def process(raw, toinclude=[0]):

                results = []

                if toinclude != [0]:

                    windows = raw.unfold(1,mks,1)

                    rolling_mean = torch.mean(windows, dim=2)
                    rolling_std  = torch.std(windows, dim=2)
                    rolling_max  = torch.max(windows, dim=2)[0]
                    rolling_min  = torch.min(windows, dim=2)[0]

                    del windows

                if 0 in toinclude: results.append(raw)
                if 1 in toinclude: results.append(rolling_mean)
                if 2 in toinclude: results.append(rolling_std)#TODO: nans?
                if 3 in toinclude: results.append(rolling_max)

                if 4 in toinclude:

                    # Inspired on https://www.investopedia.com/terms/b/bollingerbands.asp
                    rolling_deviation = (raw[:,-rolling_mean.shape[1]:] - rolling_mean) / rolling_std
                    rolling_deviation = torch_nan_to_num(rolling_deviation)
                    results.append(rolling_deviation)

                    # h = rolling_deviation.shape[1] // 2
                    # batch_signals.append( ((torch.sum((rolling_deviation[:,:h] > 5) | (rolling_deviation[:,:h] < -5),dim=1)>0)*1).cpu().numpy() )
                    # batch_signals.append( ((torch.sum((rolling_deviation[:,-h:] > 5) | (rolling_deviation[:,-h:] < -5),dim=1)>0)*1).cpu().numpy() )

                if 5 in toinclude:

                    # Inspired on https://www.investopedia.com/terms/a/atr.asp
                    rolling_minmaxspread = rolling_max / rolling_min - 1
                    results.append(torch_nan_to_num(rolling_minmaxspread))

                #TODO: con una sola linea peta
                for i in range(len(results)):

                    #TODO: copy is necessary?
                    results[i] = F.avg_pool1d(results[i].unsqueeze(0), kernel_size=kernel_size, stride=stride).squeeze(0).cpu().numpy().copy()

                    assert((~np.isfinite(results[i])).sum() == 0)

                batch_series.extend(results)

            def process_map(raws):

                #TODO: con una sola linea peta
                for i in range(4):

                    #TODO: copy is necessary?
                    raws[i] = F.avg_pool1d(raws[i].unsqueeze(0), kernel_size=kernel_size, stride=stride).squeeze(0).cpu().numpy().copy()

                    assert((~np.isfinite(raws[i])).sum() == 0)

                #TODO: sacar signals? o algo?

                batch_maps.append(np.stack(raws,axis=1))

            bid_px1 = torch.tensor(series[start:end,0], device=device)
            ask_px1 = torch.tensor(series[start:end,1], device=device)
            bid_px2 = torch.tensor(series[start:end,2], device=device)
            ask_px2 = torch.tensor(series[start:end,3], device=device)

            bid_qty1 = torch.tensor(series[start:end,4], device=device)
            ask_qty1 = torch.tensor(series[start:end,5], device=device)
            bid_qty2 = torch.tensor(series[start:end,6], device=device)
            ask_qty2 = torch.tensor(series[start:end,7], device=device)

            texecqty_mean = torch.tensor(stock_execqty_mean[start:end], device=device).unsqueeze(1)
            texeccount_mean = torch.tensor(stock_execcount_mean[start:end], device=device).unsqueeze(1)

            process_map([   ask_qty2 / texecqty_mean,
                            ask_qty1 / texecqty_mean,
                            bid_qty1 / texecqty_mean,
                            bid_qty2 / texecqty_mean    ])

            # process_map([   torch_diff(ask_qty2 / texecqty_mean),
            #                 torch_diff(ask_qty1 / texecqty_mean),
            #                 torch_diff(bid_qty1 / texecqty_mean),
            #                 torch_diff(bid_qty2 / texecqty_mean)    ])

            # Inspired on ...
            # def get_liquidityflow(prices, qties, isbuy):

            #     price_diff = torch_diff(prices)
            #     vol_diff   = torch_diff(qties)

            #     if isbuy:

            #         return  (price_diff == 0) * vol_diff + \
            #                 (price_diff > 0)  * qties[:,1:] - \
            #                 (price_diff < 0)  * torch.roll(qties, 1, 1)[:,1:]

            #     else:

            #         return  (price_diff == 0) * vol_diff - \
            #                 (price_diff > 0)  * torch.roll(qties, 1, 1)[:,1:] + \
            #                 (price_diff < 0)  * qties[:,1:]

            # OBLa2 = get_liquidityflow(ask_px2, ask_qty2 / texecqty_mean, False)
            # OBLa1 = get_liquidityflow(ask_px1, ask_qty1 / texecqty_mean, False)
            # OBLb1 = get_liquidityflow(bid_px1, bid_qty1 / texecqty_mean, True)
            # OBLb2 = get_liquidityflow(bid_px2, bid_qty2 / texecqty_mean, True)

            #process_map([OBLa2, OBLa1, OBLb1, OBLb2])

            ##########################################################################################################

            executed_qty = torch.tensor(series[start:end,9], device=device)
            order_count = torch.tensor(series[start:end,10], device=device)

            executed_qty = executed_qty / texecqty_mean
            #process(executed_qty, [0,2,3])
            
            order_count = order_count / texeccount_mean
            #process(order_count, [0,2,3])

            #process(torch_nan_to_num(executed_qty / order_count), [0,2,3])

            t_bid_size = bid_qty1 + bid_qty2
            t_ask_size = ask_qty1 + ask_qty2

            # total volume
            process(torch_diff(torch.log(t_bid_size + t_ask_size)), [2,3])
            #process(torch.log(t_bid_size + t_ask_size), [0,4,5])

            # volume unbalance
            process(torch_diff(t_ask_size / ( t_ask_size + t_bid_size )), [2,3])
            #process((ask_qty2 + bid_qty2 - ask_qty1 - bid_qty1) / texecqty_mean, [0])
            #process((ask_qty1 + bid_qty2 - ask_qty2 - bid_qty1) / texecqty_mean, [0])

            ##########################################################################################################

            WAP = (bid_px1*ask_qty1 + ask_px1*bid_qty1) / (bid_qty1 + ask_qty1)

            WAP_returns = torch_diff(torch.log(WAP))

            # past realized volatility
            past_real_vol[start:end] = torch.sqrt(torch.sum(torch.pow(WAP_returns,2),dim=1)).cpu().numpy()

            w_avg_bid_price = (bid_px1*bid_qty1 + bid_px2*bid_qty2) / t_bid_size
            w_avg_ask_price = (ask_px1*ask_qty1 + ask_px2*ask_qty2) / t_ask_size

            deepWAP = (w_avg_bid_price * t_ask_size + w_avg_ask_price * t_bid_size) / (t_bid_size + t_ask_size)
            
            midprice = (bid_px1 + ask_px1) / 2

            executed_px = torch.tensor(series[start:end,8], device=device)
            executed_px = torch_nan_to_num(executed_px) + torch.isnan(executed_px) * torch_nan_to_num(WAP)

            # # Inspired on https://www.investopedia.com/terms/v/vwap.asp

            money_flow = (executed_px * executed_qty)[:,1:]
            money_flows_windows = money_flow.unfold(1,mks,1)
            executed_qty_windows = executed_qty[:,1:].unfold(1,mks,1)

            VWAP = torch.sum(money_flows_windows, axis=2) / torch.sum(executed_qty_windows, axis=2)
            VWAP = torch.tensor(pd.DataFrame(VWAP.cpu().numpy()).ffill(axis=1).values, device=device)
            VWAP = torch_nan_to_num(VWAP) + torch.isnan(VWAP) * torch_nan_to_num(WAP[:,-VWAP.shape[1]:])  # for the beginning of the series
            VWAP_returns = torch_diff(torch.log(VWAP))

            pi = [2,4]#5 es sensible al price unit? parece que no ayuda
            process(WAP, pi)
            process(VWAP, pi)
            process(deepWAP, pi)
            process(midprice, pi)
            process(executed_px, pi)

            #returns 2 seguro; parece que 0 mejora un poco; 4 parece que muy poco
            ri = [0,2]
            process(WAP_returns,ri)
            process(VWAP_returns,ri)
            process(torch_diff(torch.log(deepWAP)),ri)
            process(torch_diff(torch.log(midprice)),ri)
            process(torch_diff(torch.log(executed_px)),ri)

            del money_flows_windows
            del executed_qty_windows

            #spreads 0 seguro; all?
            process(ask_px1 / bid_px1 - 1, [0,2,3,4])#4 parece que no ayuda mucho

            # comentar o descomentar, pero no tocar el toinclude - no esta claro si ayuda o no
            # process(deepWAP / WAP - 1, [0])
            # process(VWAP / WAP[:,-VWAP.shape[1]:] - 1, [0])
            # process(midprice / WAP - 1, [0])
            # process(executed_px / WAP - 1, [0])
            
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
            del midprice

            ##########################################################################################################

            # # Inspired on https://www.investopedia.com/terms/o/onbalancevolume.asp
            # OBV = reduce(((executed_qty[:,1:] * (torch_diff(executed_px) > 0)) - (executed_qty[:,1:] * (torch_diff(executed_px) < 0))) / texecqty_mean)

            # # Inspired on https://www.investopedia.com/terms/f/force-index.asp
            # # force_index = reduce(torch.diff(torch.log(executed_px)) * executed_qty[:,1:] * 1e5 / texecqty_mean)

            # # TODO: ??????????
            # # executed_qty_windows = executed_qty.unfold(1,m_kernel_sizes[1],1)
            # # moving_executed_qty = torch.mean(executed_qty_windows, axis=2)
            # # effort = (executed_qty[:,1:] / WAP_returns) / moving_executed_qty
            # # effort = torch.Tensor(torch.nan_to_num(effort))
            # # effort = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)(effort.unsqueeze(0)).squeeze()


            # # Inspired on https://www.investopedia.com/terms/m/mfi.asp

            # money_flow_pos = money_flow * (torch_diff(executed_px) > 0)
            # money_flow_neg = money_flow * (torch_diff(executed_px) < 0)

            # money_flow_pos_windows = money_flow_pos.unfold(1,mks,1)
            # money_flow_neg_windows = money_flow_neg.unfold(1,mks,1)

            # money_flow_pos_sums = torch.nansum(money_flow_pos_windows, axis=2)
            # money_flow_neg_sums = torch.nansum(money_flow_neg_windows, axis=2)

            # MFI = money_flow_pos_sums / (money_flow_pos_sums + money_flow_neg_sums)
            # process(torch_nan_to_num(MFI,nan=0.5))

            del money_flow
            # del money_flow_pos
            # del money_flow_neg
            # del money_flow_pos_windows
            # del money_flow_neg_windows
            # del money_flow_pos_sums
            # del money_flow_neg_sums
            # #del MFI

            # ################################################################################################################################

            #TODO: lo mismo con VWAP
            #TODO: lo mismo con WAP_returns, deepWAP_returns y midprice_returns? lo mismo con otros precios??
            WAP_windows = WAP.unfold(1,mks,1)

            WAP_rolling_mean = torch.mean(WAP_windows, axis=2)
            WAP_rolling_std  = torch.std(WAP_windows, axis=2)
            WAP_rolling_max  = torch.max(WAP_windows, axis=2).values
            WAP_rolling_min  = torch.min(WAP_windows, axis=2).values

            #process(WAP_rolling_std,[1,3,5])  #TODO: 3 y 5 son utiles?

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
            #process(torch_nan_to_num(donchian_deviation),[0,4])

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


            lengths = []

            for s in batch_series:
                # if start==0:
                #     print(s.shape[1])
                lengths.append(s.shape[1])

            for m in batch_maps:
                lengths.append(m.shape[2])

            if start==0:
                print('min length found:',min(lengths))
                print('max length found:',max(lengths))

            for i in range(len(batch_series)):

                excess = batch_series[i].shape[1] - fixed_length

                if excess < 0:
                
                    backf = np.tile(batch_series[i][:,0:1],(1,-excess))
                    batch_series[i] = np.concatenate((backf,batch_series[i]),axis=-1)
                
                else:
                    batch_series[i] = batch_series[i][:,-fixed_length:]  #TODO

            for i in range(len(batch_maps)):
                batch_maps[i] = batch_maps[i][:,:,-fixed_length:]  #TODO

            processed_series.append(np.stack(batch_series, axis=1))
            processed_maps.append(np.stack(batch_maps, axis=1))
            #signals.append(np.stack(batch_signals, axis=1))

            gc.collect()
            torch.cuda.empty_cache()

        series = np.vstack(processed_series)
        maps = np.vstack(processed_maps)
        #signals = np.vstack(signals)
        gc.collect()

        #assert((~np.isfinite(signals)).sum() == 0)

        print('series shape:\t',series.shape)
        print('maps shape:\t',maps.shape)
        #print('signals shape:\t',signals.shape)

        print('series min:',list(np.round(np.min(series,axis=(0,2)),5)))
        #print('series med:',list(np.round(np.median(series,axis=(0,2)),5)))
        print('series max:',list(np.round(np.max(series,axis=(0,2)),5)))

        print('maps min:',list(np.round(np.min(maps,axis=(0,2,3)),5)))
        #print('maps med:',list(np.round(np.median(maps,axis=(0,2,3)),5)))
        print('maps max:',list(np.round(np.max(maps,axis=(0,2,3)),5)))

        #print('signals props:',np.sum(signals,axis=0)/len(signals))

        ###############################################################################

        print('computing targets...')

        fut_rea_vol = targets[:,3]
        rea_vol_delta = fut_rea_vol - past_real_vol
        rea_vol_increase = fut_rea_vol / past_real_vol - 1

        targets = np.hstack((   targets,
                                np.expand_dims(past_real_vol,1),
                                np.expand_dims(rea_vol_delta,1),
                                np.expand_dims(rea_vol_increase,1)
                            ))

        assert((~np.isfinite(targets)).sum() == 0)

        ###############################################################################

        print('computing series stats...')

        statos = []

        r = series.shape[2]
        ds = [1]#[1, 2, 4]
        for d in ds:

            s = series[:,:,-r//d:]

            statos.append(np.hstack((

                np.mean(s,axis=2),
                #np.std(s,axis=2),#TODO: std de std en algunos casos, realmente necesario?
                #np.max(s,axis=2)
                    
             )))

        stats = []
        stats.append(execqty_sum.reshape(-1,1))
        stats.append(execcount_sum.reshape(-1,1))
        stats.append(past_real_vol.reshape(-1,1))
        stats = np.hstack(stats)

        # corrs = np.corrcoef(np.hstack((stats, np.expand_dims(targets[:,-1],1))).T)[:-1,-1]
        # print('sorted abs corrs:')
        # print(np.round(np.sort(np.abs(corrs))[::-1],2))

        #THRESHOLD = 0.
        #stats = stats[:,np.abs(corrs)>THRESHOLD]  #TODO
        print('stats shape:\t',stats.shape)

        print('coeffs:')
        print(np.round(np.corrcoef(np.hstack((np.hstack(statos), stats, rea_vol_increase.reshape(-1,1))).T)[:-1,-1],2))

        ###############################################################################

        print('scaling...')

        #series = series - np.median(series, axis=0)

        idx_train = stock_id % 5 != CV_split

        stmf = settings['PREPROCESS_DIR'] + 'stats_means_{}.npy'.format(CV_split)
        stsf = settings['PREPROCESS_DIR'] + 'stats_stds_{}.npy'.format(CV_split)
        semf = settings['PREPROCESS_DIR'] + 'series_means_{}.npy'.format(CV_split)
        sesf = settings['PREPROCESS_DIR'] + 'series_stds_{}.npy'.format(CV_split)
        mamf = settings['PREPROCESS_DIR'] + 'maps_means_{}.npy'.format(CV_split)
        masf = settings['PREPROCESS_DIR'] + 'maps_stds_{}.npy'.format(CV_split)

        if settings['ENV'] == 'train':

            series_means = np.expand_dims(np.mean(series[idx_train], axis=(0,2)),1)
            series_stds = np.expand_dims(np.std(series[idx_train], axis=(0,2)),1)

            assert((~np.isfinite(series_means)).sum() == 0)
            assert((~np.isfinite(series_stds)).sum() == 0)
            np.save(semf, series_means)
            np.save(sesf, series_stds)
        else:

            series_means = np.load(semf)
            series_stds  = np.load(sesf)

        series = (series - series_means) / series_stds

        # if settings['ENV'] == 'train':

        #     maps_means = np.expand_dims(np.mean(maps[idx_train], axis=(0,2,3)),1)
        #     maps_stds = np.expand_dims(np.std(maps[idx_train], axis=(0,2,3)),1)

        #     assert((~np.isfinite(maps_means)).sum() == 0)
        #     assert((~np.isfinite(maps_stds)).sum() == 0)
        #     np.save(mamf, maps_means)
        #     np.save(masf, maps_stds)
        # else:

        #     maps_means = np.load(mamf)
        #     maps_stds  = np.load(masf)

        # maps = (maps - np.expand_dims(maps_means,1)) / np.expand_dims(maps_stds,1)

        assert((~np.isfinite(series)).sum() == 0)
        assert((~np.isfinite(stats)).sum() == 0)
        
        self.series = series
        self.maps = maps
        #self.signals = signals
        self.stats = stats
        self.targets = targets

        print('computing series medians...')
        self.series_medians = series_means#np.median(series, axis=0)  #TODO
        #TODO: maps medians?

        # stock_pastvol_mean = np.repeat(np.nan, len(series)).astype(np.float32)

        # for i in np.unique(stock_id):
        #     idx = stock_id == int(i)
        #     stock_pastvol_mean[idx] = np.mean(past_real_vol[idx])



    def train_dataloader(self):

        #TODO: sorting por volatilidad
        idx = self.targets[:,0] % 5 != self.CV_split
        return DataLoader(SeriesDataSet(self.series[idx], self.maps[idx], self.stats[idx], self.targets[idx]), batch_size=512, shuffle=True)

    def val_dataloader(self):

        idx = self.targets[:,0] % 5 == self.CV_split
        return DataLoader(SeriesDataSet(self.series[idx], self.maps[idx], self.stats[idx], self.targets[idx]), batch_size=1024, shuffle=False)



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



class SeriesDataSet(Dataset):
    
    def __init__(self, series, maps, stats, targets):

        super(SeriesDataSet, self).__init__()

        self.series = series
        self.maps = maps
        self.stats = stats
        self.targets = targets

    def __len__(self):

        return len(self.series)

    def __getitem__(self, idx):

        return self.series[idx], self.maps[idx], self.stats[idx], self.targets[idx]



if __name__ == '__main__':

    data = OptiverDataModule()
