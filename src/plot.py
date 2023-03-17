# impprt libraries
from typing import Optional
from datetime import timedelta
import pandas as pd

import plotly.express as px

# creating a plot with the function

### OPTION 2 ###
def plot_one_sample(
        features: pd.DataFrame,
        target: pd.Series,
        example_id: int,
        predictions: Optional[pd.Series] = None,
):
    """
    Plot a single sample with its target and prediction.
    """

    # get the feature
    features_ = features.iloc[example_id]

    # get the target
    target_ = target.iloc[example_id]

    ts_columns = [c for c in features.columns if c.startswith("rides_previous_")]
    ts_values = [features_[c] for c in ts_columns] + [target_]
    ts_dates = pd.date_range(
        features_['pickup_hour'] - timedelta(hours=len(ts_columns)),
        features_['pickup_hour'],
        freq='H'
    )

    # line plot with past values
    title = f'Pick up hour: {features_["pickup_hour"]}, location_id={features_["pickup_location_id"]}'
    fig = px.line(
        x=ts_dates,
        y=ts_values,
        template='plotly_dark',
        markers=True,
        title=title,
        labels={'x': 'Time', 'y': 'Rides'}
    )

    # green dot for the value to predict
    fig.add_scatter(
        x=ts_dates[-1:], y=[target_],
        line_color='green',
        mode='markers', marker_size=10, name='actual value' 
    )

    if predictions is not None:
        # big red X for the predicted value, if passed
        prediction_ = predictions.iloc[example_id]
        fig.add_scatter(
            x=ts_dates[-1:], y=[prediction_],
            line_color='red',
            mode='markers', marker_symbol='x', marker_size=25,
            name='prediction'
        )
    return fig    

### OPTION 2 ###