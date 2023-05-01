# Model trained on 8 weeks data

import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline

import lightgbm as lgb

def average_rides_last_8_weeks(X: pd.DataFrame) -> pd.DataFrame:
    """
    Adds one column with the average rides from
    - 7 days ago
    - 14 days ago
    - 21 days ago
    - 28 days ago
    - 35 days ago
    - 42 days ago
    - 49 days ago
    - 56 days ago

    Parameters
    ----------
    X : pd.DataFrame
        Input data

    Returns
    -------
    pd.DataFrame
        Input data with one additional column      
    """
    X['average_rides_last_8_weeks'] = 0.125*(
        X[f'rides_previous_{7*24}_hour'] + \
        X[f'rides_previous_{2*7*24}_hour'] + \
        X[f'rides_previous_{3*7*24}_hour'] + \
        X[f'rides_previous_{4*7*24}_hour'] + \
        X[f'rides_previous_{5*7*24}_hour'] + \
        X[f'rides_previous_{6*7*24}_hour'] + \
        X[f'rides_previous_{7*7*24}_hour'] + \
        X[f'rides_previous_{8*7*24}_hour']
    )
    return X


class TemporalFeaturesEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn data transformation that adds 2 columns
    - hour
    - day_of_week
    and removes the `pickup_hour` datetime column.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        X_ = X.copy()
        
        # Generate numeric columns from datetime
        X_["hour"] = X_['pickup_hour'].dt.hour
        X_["day_of_week"] = X_['pickup_hour'].dt.dayofweek
        
        return X_.drop(columns=['pickup_hour'])

def get_pipeline(**hyperparams) -> Pipeline:

    # sklearn transform
    add_feature_average_rides_last_8_weeks = FunctionTransformer(
        average_rides_last_8_weeks, validate=False)
    
    # sklearn transform
    add_temporal_features = TemporalFeaturesEngineer()

    # sklearn pipeline
    return make_pipeline(
        add_feature_average_rides_last_8_weeks,
        add_temporal_features,
        lgb.LGBMRegressor(**hyperparams)
    )