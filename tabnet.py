import torch
import numpy as np

from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import StandardScaler

from data import OptiverDataModule


if __name__ == '__main__':

    data = OptiverDataModule()

    idx_train = data.targets[:,0] % 5 != 0
    idx_valid = data.targets[:,0] % 5 == 0

    X_train = np.hstack((data.signals[idx_train], data.stats[idx_train]))
    y_train = (data.targets[idx_train,-1]>0)*1

    X_valid = np.hstack((data.signals[idx_valid], data.stats[idx_valid]))
    y_valid = (data.targets[idx_valid,-1]>0)*1

    torch.manual_seed(0)
    np.random.seed(0)

    clf = TabNetClassifier( n_d=32,
                            n_a=32,
                            n_steps=3,
                            cat_idxs=list(range(data.signals.shape[1])),
                            cat_dims=[2]*data.signals.shape[1],
                            cat_emb_dim=1,
                            optimizer_fn=torch.optim.Adam,
                            optimizer_params=dict(lr=2e-2),
                            scheduler_params={"step_size":5, "gamma":0.8},
                            scheduler_fn=torch.optim.lr_scheduler.StepLR )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train),(X_valid, y_valid)],
        eval_name=['train','valid'],
        batch_size = 4096,
        virtual_batch_size = 4096//8,
        patience = 7,
        max_epochs = 100
    )

    print('importances:',clf.feature_importances_*100)

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
