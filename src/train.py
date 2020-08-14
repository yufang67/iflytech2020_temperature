import os
import pandas as pd
import numpy as np
import joblib
import argparse

import config
import model_dispatcher

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.metrics import make_scorer

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor


def run(model):
    trainX = pd.read_csv(config.TRAIN_X)
    trainY = pd.read_csv(config.TRAIN_Y)

    nums = int(trainX.shape[0] * 0.5)
    trn_x, trn_y, val_x, val_y = trainX[:nums], trainY[:nums], trainX[nums:], trainY[nums:]
    if model=='lgb':
        
        train_matrix = lgb.Dataset(trn_x, label=trn_y)
        valid_matrix = lgb.Dataset(val_x, label=val_y)
        data_matrix  = lgb.Dataset(trainX, label=trainY)

        param_lgb = model_dispatcher.models[model]
        model_lgb = lgb.train(param_lgb, train_matrix, 50000, valid_sets=[train_matrix, valid_matrix],verbose_eval=500,early_stopping_rounds=1000)
        ml = lgb.train(param_lgb, data_matrix, model_lgb.best_iteration)
        print('model_best_iteration:', model_lgb.best_iteration)
        save_to = config.MODEL_OUTPUT + '{}.txt'.format(model)
        ml.save_model(save_to,num_iteration = model_lgb.best_iteration )

    elif model=='xgb':

        train_matrix = xgb.DMatrix(trn_x.values , label=trn_y.values, missing=np.nan)
        valid_matrix = xgb.DMatrix(val_x.values , label=val_y.values, missing=np.nan)
        data_matrix  = xgb.DMatrix(trainX.values, label=trainY.values, missing=np.nan)

        watchlist = [(train_matrix, 'train'),(valid_matrix, 'eval')]
        param_xgb = model_dispatcher.models[model]
        
        ml = xgb.train(param_xgb, data_matrix, num_boost_round=50000, evals=watchlist, verbose_eval=500, early_stopping_rounds=1000)
        print('model_best_ntree:', ml.best_ntree_limit)
        ml.save_model(os.path.join(config.MODEL_OUTPUT,"{}.model".format(model)))
    
    elif model=='catb':
        param_catb = model_dispatcher.models[model]
        
        ml = CatBoostRegressor(iterations=20000, **param_catb)
        ml.fit(trainX.values, trainY.values, eval_set=(val_x.values, val_y.values),cat_features=[], use_best_model=True, verbose=500)

        ml.save_model(os.path.join(config.MODEL_OUTPUT,"{}.model".format(model)))
    else:
        model_val = model_dispatcher.models[model]
        model_val.fit(trn_x.values,trn_y.values.ravel())
        val_pred = model_val.predict(val_x.values)
        
        mse = mean_squared_error(val_y,val_pred)
        abe = mean_absolute_error(val_y,val_pred)

        print('mse_val:',mse)
        print('abs_val:',abe)

        ml = model_dispatcher.models[model]
        ml.fit(trainX.values,trainY.values.ravel())


        joblib.dump(
            ml,
            os.path.join(config.MODEL_OUTPUT,"{}.bin".format(model))
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", 
        type=str
    )

    args = parser.parse_args()

    run(
        model = args.model
    )
