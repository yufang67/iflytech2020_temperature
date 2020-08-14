import pandas as pd
import numpy as np

import lightgbm as lgb
from functools import partial


from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold, KFold,TimeSeriesSplit
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error

import optuna
import config

    


def optimize(trial,x,y):
    param = {
        #'boosting_type': 'gbdt',
        'objective': 'mse',
        'learning_rate' : trial.suggest_loguniform('learning_rate', 1e-4,1e-1),
        'n_estimators' : trial.suggest_int('n_estimators',100,5000),
        'max_depth' : trial.suggest_int('max_depth', 3, 10),
        'num_leaves' : trial.suggest_int('num_leaves', 2, 2**10),
        'max_bin' : trial.suggest_int('max_bin', 10, 1000), 
        'feature_fraction' : trial.suggest_uniform('feature_fraction', 0, 1),
        'bagging_fraction' : trial.suggest_uniform('bagging_fraction', 0, 1),    
        'bagging_freq' : trial.suggest_int('bagging_freq', 1, 100),
        'min_sum_hessian_in_leaf' : trial.suggest_int('bagging_freq', 1, 10),
        'reg_alpha' : trial.suggest_loguniform('reg_alpha', 1e-5, 1e-1),
        'reg_lambda' : trial.suggest_loguniform('reg_lambda', 1e-5, 1e-1),
    #'nthread': -1,
    #'device': 'gpu',
    #'gpu_platform_id': 0,
    #'gpu_device_id': 0
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

        train_matrix = lgb.Dataset(xtrain, label=ytrain)
        valid_matrix = lgb.Dataset(xtest, label=ytest)
        
        model = lgb.train(param, train_matrix, valid_sets=[train_matrix, valid_matrix],verbose_eval=500,early_stopping_rounds=1000)

        val_pred  = model.predict(xtest, num_iteration=model.best_iteration)
        
        fold_mse = mean_squared_error(ytest,val_pred)
        mse.append(fold_mse)

    return np.mean(mse)


if __name__ == "__main__":
    trainX = pd.read_csv(config.TRAIN_X).values
    #trainY = pd.read_csv(config.TRAIN_Y).values.squeeze()
    trainY = pd.read_csv(config.TRAIN_T).values.squeeze()

    #print(trainY,type(trainY),trainY.shape)
    optimization_function = partial(optimize, x=trainX, y=trainY)
    study = optuna.create_study(direction="minimize")
    study.optimize(optimization_function,n_trials=100)
    print('######################')
    print(study.best_params)

###############################################################


