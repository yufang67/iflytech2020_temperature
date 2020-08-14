import pandas as pd
import numpy as np
import config

from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge, Lasso
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor


from mlxtend.regressor import StackingCVRegressor

pred_lgb =  pd.read_csv("../submission/subV10_lgb.csv")
pred_xgb = pd.read_csv("../submission/subV15_xgb.csv")
pred_rf = pd.read_csv("../submission/subV11_rf.csv")
pred_lasso = pd.read_csv("../submission/subV19_lasso.csv")
pred_xtr = pd.read_csv("../submission/subV22_xtr.csv")


lb = [
    0.1338, 
    0.13617, 
    0.14128, 
    0.13521, 
    #0.14446
    ]
weight = 1 - lb/np.sum(lb)
weight = weight/np.sum(weight)
#print(weight)
pred_ls = [
    pred_lgb, 
    pred_xgb, 
    pred_rf, 
    pred_lasso, 
    #pred_xtr
    ]
pred = pred_lgb.copy()
#print(type(pred))

pred['temperature'] = np.zeros(len(pred['time'].values))

#print(pred)


for i in range(len(weight)):
    pred['temperature'] = pred['temperature'].values + weight[i] * pred_ls[i]['temperature'].values

#print(pred)

#pred = (pred_lgb + pred_xgb + pred_rf)/3


#stack_gen = StackingCVRegressor(regressors=(xgb, lgbm, svr, rg, rf),
#                                meta_regressor=lgbm,
#                                use_features_in_secondary=True,n_jobs=-1)


pred.to_csv('../submission/sub_en.csv', index=False)
print(pred.shape)
print(pred.isnull().any())
#print(pred_lgb.shape)
#pred.head()




