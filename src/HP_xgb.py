import pandas as pd
import numpy as np

import xgboost as xgb
from functools import partial


from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold, KFold,TimeSeriesSplit
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error

import optuna
import config


def optimize(trial,x,y):
    param = {
        'booster': 'gbtree',
        'eval_metric': 'rmse',
        'seed': 2020,
        'nthread': -1,
        'silent': True,
        'learning_rate' : trial.suggest_loguniform('learning_rate', 1e-3,1e-1),
        #'n_estimators' : trial.suggest_int('n_estimators',100,5000),
        'max_depth' : trial.suggest_int('max_depth', 3, 15),
        'gamma' : trial.suggest_loguniform('gamma',1e-3,1),
        'subsample' : trial.suggest_uniform('subsample', 0, 1),
        'colsample_bytree' : trial.suggest_uniform('colsample_bytree', 0, 1),
        'min_child_weight' : trial.suggest_uniform('min_child_weight', 1e-2, 10),
        'reg_alpha' : trial.suggest_loguniform('reg_alpha', 1e-4, 1e-1),

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

        train_matrix = xgb.DMatrix(xtrain , label=ytrain, missing=np.nan)
        valid_matrix = xgb.DMatrix(xtest , label=ytest, missing=np.nan)
        watchlist = [(train_matrix, 'train'),(valid_matrix, 'eval')]
        
        model = xgb.train(param, train_matrix, 50000, evals=watchlist, verbose_eval=500, early_stopping_rounds=1000)

        val_pred  = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit)
        
        fold_mse = mean_squared_error(ytest,val_pred)
        mse.append(fold_mse)

    return np.mean(mse)


if __name__ == "__main__":
    trainX = pd.read_csv(config.TRAIN_X).values
    trainY = pd.read_csv(config.TRAIN_Y).values

    optimization_function = partial(optimize, x=trainX, y=trainY)
    study = optuna.create_study(direction="minimize")
    study.optimize(optimization_function,n_trials=50)
    print('######################')
    print(study.best_params)

###############################################################
#  {'learning_rate': 0.001961744302808936, 
#'n_estimators': 1264, 
#'max_depth': 3, 
#'gamma': 0.10609220143686166, 
#'subsample': 0.5539858308859825, 
#'colsample_bytree': 0.8108188113308883, 
#'min_child_weight': 0.3468511768465383, 
#'reg_alpha': 0.09725427660294955}. 
#Best is trial#13 with value: 0.10005959928970835.




