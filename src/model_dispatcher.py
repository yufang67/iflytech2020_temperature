from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge, Lasso
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, ExtraTreesRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor



param_rg = {
    'alpha':1.9,
    'solver': 'sparse_cg',
    'fit_intercept': False,
    'normalize': True,
    'random_state':42,
}
#'alpha': 1.90, 'solver': 'sparse_cg', 
#'fit_intercept': False, 'normalize': True

param_lasso = {
    'alpha':0.0001,
    'normalize':True, 
    'random_state':2020
}


param_rf = {
    'n_estimators':238,
    'max_depth':34,
    'min_samples_split':8,
    'min_samples_leaf':15,
    'n_jobs':-1,
    'random_state':2020,
}
#'n_estimators': 194, 'max_depth': 46, 'min_samples_split': 100, 'min_samples_leaf': 3, 'max_features': None
param_xtr = {
   'max_depth':10, 
   'max_features':'log2', 
   'min_samples_leaf':2,
   'min_samples_split':2, 
   'n_estimators':500,
   'n_jobs':-1,
   'random_state':42,
}
#'n_estimators': 452, 'max_depth': 21, 'min_samples_split': 96, 'min_samples_leaf': 5, 'max_features': None
'''
param_baye = {
            'n_iter':10,
            'alpha': 1e4,
            'alpha_1':1e-3,
            'alpha_2':1e-3,
            'lambda_1':1e-3,
            'lambda_2':1e-3,
            }
'''
param_svm={
    'kernel': 'linear',
    'C' : 0.13673,
    'gamma': 'scale',
}
#'kerne': 'linear', 'C': 0.13673, 'gamma': 'scale'

param_lgb = {
    #'boosting_type': 'gbdt',
    'objective': 'mse',
    'learning_rate': 0.043,
    'n_estimators': 1936,
    'max_depth': 4,
    'num_leaves':2,
    'max_bin': 90,
    'feature_fraction': 0.45,
    'bagging_fraction': 0.75,
    'bagging_freq': 4,
    'seed': 2020,
    'reg_alpha': 0.000028, 
    'reg_lambda': 0.00037,
    #'nthread': -1,
    #'device': 'gpu',
    #'gpu_platform_id': 0,
    #'gpu_device_id': 0
    }
#'learning_rate': 0.043, 'n_estimators': 1936, 'max_depth': 4, 'num_leaves': 2, 'max_bin': 90, 
#'feature_fraction': 0.45, 'bagging_fraction': 0.75, 'bagging_freq': 4, 'reg_alpha': 0.00028, 'reg_lambda': 0.00037

#'learning_rate': 0.059, 'n_estimators': 1287, 'max_depth': 3, 'num_leaves': 507, 'max_bin': 65, 
#'feature_fraction': 0.58, 'bagging_fraction': 0.545, 'bagging_freq': 33, 'reg_alpha': 9.289706778377865e-05, 'reg_lambda': 0.0044

param_xgb = {'booster': 'gbtree',
'eval_metric': 'rmse',
'learning_rate': 0.00196,
'n_estimators': 1264,
'max_depth': 3,
'gamma': 0.10609,
'min_child_weight': 0.347,
'subsample': 0.554,
'colsample_bytree': 0.81,
'reg_alpha': 0.0972,
#'eta': 0.001,
'seed': 2020,
'nthread': -1,
'silent': True,
}
#'learning_rate': 0.027, 'n_estimators': 1860, 'max_depth': 4, 'gamma': 0.29, 
#'subsample': 0.048, 'colsample_bytree': 0.72, 'min_child_weight': 0.25, 'reg_alpha': 0.0001

#'learning_rate': 0.0226, 'max_depth': 5, 'gamma': 0.034, 'subsample': 0.0286, 
#'colsample_bytree': 0.82, 'min_child_weight': 2.700, 'reg_alpha': 0.019

param_catb = {
    'learning_rate': 0.001, 
    'depth': 5, 
    'l2_leaf_reg': 10, 
    'bootstrap_type': 'Bernoulli',
    'od_type': 'Iter', 
    'od_wait': 50, 
    'random_seed': 11, 
    'allow_writing_files': False
    }
#'bootstrap_type': 'Bayesian', 'learning_rate': 0.058, 'depth': 3, 
#'od_type': 'IncToDec', 'l2_leaf_reg': 0.00015


models = {
    'rg':Ridge(**param_rg),
    'lasso':Lasso(**param_lasso),
 #   'baye':BayesianRidge(**param_baye),
    'svm':SVR(**param_svm),
    'rf':RandomForestRegressor(**param_rf),
    'xtr':ExtraTreesRegressor(**param_xtr),
    'lgb':param_lgb,
    'xgb':param_xgb,
    'catb':CatBoostRegressor(iterations=20000, **param_catb),
    
}
