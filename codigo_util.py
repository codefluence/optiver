

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



        # sliding_window_view not available for numpy < 1.20
        # moving_realized_volatility = np.apply_along_axis(lambda x: np.sqrt(np.sum(x**2)), arr=windows, axis=2)



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




        # for i in range(series.shape[1]):

        #     scaler = MinMaxScaler().fit(series[:,i].T)
        #     series[:,i] = scaler.transform(series[:,i].T).T