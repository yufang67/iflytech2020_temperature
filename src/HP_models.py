import pandas as pd
import numpy as np
import argparse

from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge, Lasso
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR

from functools import partial

#from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold, KFold,TimeSeriesSplit
#from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error

import optuna
import config

    


def optimize(trial,x,y,model):
    cv = TimeSeriesSplit(n_splits=10,max_train_size=12400)
    mse = []

    if model == 'Ridge':
        param = {
            'alpha': trial.suggest_loguniform('alpha',1e-4,1e4),
            'solver': trial.suggest_categorical('solver',['svd','cholesky','lsqr','sparse_cg','sag','saga']), 
            'fit_intercept': trial.suggest_categorical('fit_intercept',[True,False]),
            'normalize': trial.suggest_categorical('normalize',[True,False]),
            'random_state': 2020, 
            }

        for idx in cv.split(x,y):

            train_idx, test_idx = idx[0], idx[1]
            xtrain = x[train_idx]
            ytrain = y[train_idx]

            xtest = x[test_idx]
            ytest = y[test_idx]

            ml = Ridge(**param)
            ml.fit(xtrain, ytrain)
            val_pred  = ml.predict(xtest)
     
            fold_mse = mean_squared_error(ytest,val_pred)
            mse.append(fold_mse)

    elif model == 'Lasso': 
        param = {
            'alpha': trial.suggest_loguniform('alpha',1e-4,1e4),
            'normalize': trial.suggest_categorical('normalize',[True,False]),
            'random_state': 2020, 
            }

        for idx in cv.split(x,y):

            train_idx, test_idx = idx[0], idx[1]
            xtrain = x[train_idx]
            ytrain = y[train_idx]

            xtest = x[test_idx]
            ytest = y[test_idx]

            ml = Lasso(**param)
            ml.fit(xtrain, ytrain)
            val_pred  = ml.predict(xtest)
     
            fold_mse = mean_squared_error(ytest,val_pred)
            mse.append(fold_mse)

    elif model == 'baye':
        param = {
            'n_iter':trial.suggest_int('n_iter',10,1000),
            'alpha_1':trial.suggest_loguniform('alpha_1',1e-10,1e-3),
            'alpha_2':trial.suggest_loguniform('alpha_2',1e-10,1e-3),
            'lambda_1':trial.suggest_loguniform('lambda_1',1e-10,1e-3),
            'lambda_2':trial.suggest_loguniform('lambda_2',1e-10,1e-3),
            }

        for idx in cv.split(x,y):

            train_idx, test_idx = idx[0], idx[1]
            xtrain = x[train_idx]
            ytrain = y[train_idx]

            xtest = x[test_idx]
            ytest = y[test_idx]

            ml = BayesianRidge(**param)
            ml.fit(xtrain, ytrain)
            val_pred  = ml.predict(xtest)
     
            fold_mse = mean_squared_error(ytest,val_pred)
            mse.append(fold_mse)
    
    elif model == 'rf':
        param = {
            'n_estimators':trial.suggest_int('n_estimators',5,1000),
            'max_depth':trial.suggest_int('max_depth',2,50),
            'min_samples_split':trial.suggest_int('min_samples_split',1,100),
            'min_samples_leaf':trial.suggest_int('min_samples_leaf',1,10),
            'max_features':trial.suggest_categorical('max_features',['log2','sqrt',None]),
            }

        for idx in cv.split(x,y):

            train_idx, test_idx = idx[0], idx[1]
            xtrain = x[train_idx]
            ytrain = y[train_idx]

            xtest = x[test_idx]
            ytest = y[test_idx]

            ml = RandomForestRegressor(**param)
            ml.fit(xtrain, ytrain)
            val_pred  = ml.predict(xtest)
     
            fold_mse = mean_squared_error(ytest,val_pred)
            mse.append(fold_mse)

    elif model == 'xtr':
        param = {
            'n_estimators':trial.suggest_int('n_estimators',5,1000),
            'max_depth':trial.suggest_int('max_depth',2,50),
            'min_samples_split':trial.suggest_int('min_samples_split',1,100),
            'min_samples_leaf':trial.suggest_int('min_samples_leaf',1,10),
            'max_features':trial.suggest_categorical('max_features',['log2','sqrt',None]),
            'random_state':2020,
            }

        for idx in cv.split(x,y):

            train_idx, test_idx = idx[0], idx[1]
            xtrain = x[train_idx]
            ytrain = y[train_idx]

            xtest = x[test_idx]
            ytest = y[test_idx]

            ml = ExtraTreesRegressor(**param)
            ml.fit(xtrain, ytrain)
            val_pred  = ml.predict(xtest)
     
            fold_mse = mean_squared_error(ytest,val_pred)
            mse.append(fold_mse)
    elif model == 'svm':
        param = {
            'kernel':trial.suggest_categorical('kerne',['linear','poly','rbf']),
            'C': trial.suggest_loguniform('C',1e-4,1e3),
            'gamma':trial.suggest_categorical('gamma',['scale','auto'])
            }

        for idx in cv.split(x,y):

            train_idx, test_idx = idx[0], idx[1]
            xtrain = x[train_idx]
            ytrain = y[train_idx]

            xtest = x[test_idx]
            ytest = y[test_idx]

            ml = SVR(**param)
            ml.fit(xtrain, ytrain)
            val_pred  = ml.predict(xtest)
            fold_mse = mean_squared_error(ytest,val_pred)
            mse.append(fold_mse)

    return np.mean(mse)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        type=str
    )
    args = parser.parse_args()


    trainX = pd.read_csv(config.TRAIN_X).values
    trainY = pd.read_csv(config.TRAIN_Y).values.squeeze()

    #print(trainY,type(trainY),trainY.shape)
    optimization_function = partial(optimize, x=trainX, y=trainY, model=args.model)
    study = optuna.create_study(direction="minimize")
    study.optimize(optimization_function,n_trials=100)
    print('######################')
    print(study.best_params)

###############################################################


