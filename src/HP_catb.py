import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from functools import partial


from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold, KFold,TimeSeriesSplit
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error

import optuna
import config

    


def optimize(trial,x,y):
    param = {
        'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bernoulli','Bayesian']),
        'learning_rate' : trial.suggest_loguniform('learning_rate', 1e-4,1e-1),
        'depth' : trial.suggest_int('depth', 3, 10),
        #'max_leaves': trial.suggest_int('max_leaves', 2, 2**10),
        'od_type': trial.suggest_categorical('od_type',['Iter','IncToDec']),
        'l2_leaf_reg' : trial.suggest_loguniform('l2_leaf_reg', 1e-5, 1e-1),
        'random_seed': 2020, 
        'allow_writing_files': False
    }
    
    cv = TimeSeriesSplit(n_splits=10,max_train_size=12400)
    mse = []
    #abe = []

    for idx in cv.split(x,y):

        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]

        xtest = x[test_idx]
        ytest = y[test_idx]

        model = CatBoostRegressor(iterations=20000, **param)
        model.fit(xtrain, ytrain, eval_set=(xtest, ytest),cat_features=[], use_best_model=True, verbose=500)
        val_pred  = model.predict(xtest)
     
        fold_mse = mean_squared_error(ytest,val_pred)
        mse.append(fold_mse)

    return np.mean(mse)


if __name__ == "__main__":
    trainX = pd.read_csv(config.TRAIN_X).values
    trainY = pd.read_csv(config.TRAIN_Y).values.squeeze()

    #print(trainY,type(trainY),trainY.shape)
    optimization_function = partial(optimize, x=trainX, y=trainY)
    study = optuna.create_study(direction="minimize")
    study.optimize(optimization_function,n_trials=100)
    print('######################')
    print(study.best_params)

###############################################################


