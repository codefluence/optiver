import torch
import numpy as np

#TabNetClassifier
#TabNetRegressor
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from pytorch_tabnet.metrics import Metric

from data import OptiverDataModule

def rmspe_loss(y_pred, y_true):

    return torch.mean(torch.sqrt(torch.mean(torch.square((y_true - y_pred) / y_true), dim=0)))

class rmspe(Metric):

    def __init__(self):
        self._name = "rmspe"
        self._maximize = False

    def __call__(self, y_true, y_score):

        return np.mean(np.sqrt(np.mean(np.square((y_true - y_score) / y_true), axis=0)))

if __name__ == '__main__':

    data = OptiverDataModule()

    stock_id     = data.targets[:,0]
    vol_increase = data.targets[:,-1]
    fut_rea_vol  = data.targets[:,-4]

    idx_train = stock_id % 5 != 0
    idx_valid = stock_id % 5 == 0

    X_train = np.hstack((data.signals[idx_train], data.stats[idx_train])) #, np.load('./pred_train.npy').reshape(-1,1)
    X_train = data.signals[idx_train]
    y_train = (vol_increase[idx_train]>0)*1
    y_train = fut_rea_vol[idx_train].reshape(-1,1)

    X_valid = np.hstack((data.signals[idx_valid], data.stats[idx_valid]))
    X_valid = data.signals[idx_valid]
    y_valid = (vol_increase[idx_valid]>0)*1
    y_valid = fut_rea_vol[idx_valid].reshape(-1,1)

    torch.manual_seed(0)
    np.random.seed(0)

    clf = TabNetRegressor( n_d=32,
                            n_a=32,
                            n_steps=3,
                            cat_idxs=list(range(data.signals.shape[1])),
                            cat_dims=[2]*data.signals.shape[1],
                            cat_emb_dim=1,
                            optimizer_fn=torch.optim.Adam,
                            optimizer_params=dict(lr=1e-2),
                            scheduler_params={"step_size":5, "gamma":0.8},
                            scheduler_fn=torch.optim.lr_scheduler.StepLR )
                            #scheduler_params={"mode":'min', "patience":3, "factor":1/4, "verbose":True, "min_lr":1e-6},
                            #scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train),(X_valid, y_valid)],
        eval_name=['train','valid'],
        batch_size = 4096,
        virtual_batch_size = 4096//8,
        patience = 12,
        max_epochs = 100,
        loss_fn=rmspe_loss,
        eval_metric=['mae','rmspe'],
    )

    print('importances:',clf.feature_importances_*100)

    am = clf.feature_importances_.argsort()
    print(am)
    print(clf.feature_importances_[am]*100)

    # np.save('./pred_train.npy', clf.predict_proba(X_train)[:,1])
    # np.save('./pred_valid.npy', clf.predict_proba(X_valid)[:,1])

    # selected = np.where(clf.feature_importances_*100 > 8)[0]
    # print('multiplying by:',i)
    # print('selected:',selected)
    # print('percents:',clf.feature_importances_[selected]*100)
    # print()

    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

    # clf = TabNetClassifier( n_d=8,
    #                         n_a=8,
    #                         n_steps=3,
    #                         optimizer_fn=torch.optim.Adam,
    #                         optimizer_params=dict(lr=2e-2),
    #                         scheduler_params={"step_size":6, "gamma":0.9},
    #                         scheduler_fn=torch.optim.lr_scheduler.StepLR )

    # clf.fit(
    #     X_t, y_t,
    #     eval_set=[(X_t, y_t),(X_v, y_v)],
    #     eval_name=['train','val'],
    #     batch_size = 4096,
    #     virtual_batch_size = 4096//8,
    #     patience = 7,
    #     max_epochs = 16
    # )
