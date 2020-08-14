import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from mlxtend.regressor import StackingCVRegressor

import pandas as pd
import config
import os
import argparse
import joblib


def run(model,iter=10000):
    testX = pd.read_csv(config.TEST_X)
    test_df = pd.read_csv(config.TEST)
    sub = pd.DataFrame(test_df['time'])

    if model=='lgb':
        
        MODEL_PATH = os.path.join(config.MODEL_OUTPUT,"{}.txt".format(model))
        ml = lgb.Booster(model_file=MODEL_PATH)
        pred = ml.predict(testX.values)
        
    elif model=='xgb':

        test_matrix  = xgb.DMatrix(testX.values)

        ml = xgb.Booster({'nthread':-1})
        
        MODEL_PATH = os.path.join(config.MODEL_OUTPUT,"{}.model".format(model))
        
        ml.load_model(MODEL_PATH)
        pred = ml.predict(test_matrix)
        
        
    elif model=='catb':
        
        MODEL_PATH = os.path.join(config.MODEL_OUTPUT,"{}.cbm".format(model))
        ml=CatBoostRegressor()
        ml.load_model(MODEL_PATH)
        pred = ml.predict(testX.values)
    else:
        MODEL_PATH = os.path.join(config.MODEL_OUTPUT,"{}.bin".format(model))
        ml = joblib.load(MODEL_PATH)
        pred = ml.predict(testX.values)
    
    
    #print(pred)
    #print(test_df.iloc[:,7].values)
    pred_f =  pred + test_df.iloc[:,7].values
    sub["temperature"] =  pred_f
    sub.to_csv(os.path.join(config.SUBMISSION,"{}.csv".format(model)), index=False)
    print(sub.head())
    print(sub.isna().any())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", 
        type=str
    )

    parser.add_argument(
        "--iter", 
        type=int
    )

    args = parser.parse_args()

    run(
        model = args.model,
        iter = args.iter
    )



