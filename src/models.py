from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost.sklearn import XGBRegressor
from m5py import M5Prime


def xgb(params):
    """Creates a XGBoost model with the given parameters."""
    booster, n_estimators, min_child_weight, subsample, colsample_bytree, max_depth = params
    model = XGBRegressor(booster=booster, colsample_bytree=colsample_bytree, 
                         max_depth=max_depth, min_child_weight=min_child_weight, n_estimators=n_estimators,
                         n_jobs=-1, subsample=subsample)
    return model
    

def rf(params):
    """Creates a random forest model with the given parameters."""
    n_stimators_value, max_depth_value, min_samples_split_value, min_samples_leaf_value = params

    model = RandomForestRegressor(criterion='squared_error', n_jobs=-1, n_estimators=n_stimators_value,
                                  max_depth=max_depth_value, min_samples_split=min_samples_split_value,
                                  min_samples_leaf=min_samples_leaf_value)
    return model


def m5p(params):
    """Creates a M5Prime model with the given parameters"""
    max_depth_value, smoothing_constant_value = params

    uni_model = M5Prime(criterion='squared_error', 
                    max_depth=max_depth_value,
                    smoothing_constant=smoothing_constant_value)
    
    model = MultiOutputRegressor(uni_model, n_jobs=-1)
    return model


model_set = {
    "xgb": xgb,
    "rf": rf,
    "m5p": m5p
    
}


def create_model_ml(model_name, params):
    assert model_name in model_set.keys(), "Model '{}' not supported".format(
        model_name
    )
    return model_set[model_name](params)
