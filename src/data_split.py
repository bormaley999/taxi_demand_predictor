from datetime import datetime
from typing import Tuple

import pandas as pd

def train_test_split(
        df: pd.DataFrame,
        cut_off_date: datetime,
        target_column_name: str,        
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split the data into train and test sets.
    
    Parameters
    ----------
    df : pd.DataFrame
        The data to split.
    cut_off_date : datetime
        The date to split the data on.
    target_column_name : str
        The name of the target column.
        
    Returns
    -------
    Tuple
        The train and test sets.
    """
    train_data = df[df.pickup_hour < cut_off_date].reset_index(drop=True)
    test_data = df[df.pickup_hour >= cut_off_date].reset_index(drop=True)

    X_train = train_data.drop(columns=[target_column_name])
    y_train = train_data[target_column_name]
    X_test = test_data.drop(columns=[target_column_name])
    y_test = test_data[target_column_name]

    return X_train, y_train, X_test, y_test