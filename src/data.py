# data path and import 
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
import tqdm

import pandas as pd
import numpy as np
import requests

from src.paths import RAW_DATA_DIR, TRANSFORMED_DATA_DIR, PARENT_DIR, DATA_DIR

# download one file of raw data
def download_one_file_of_raw_data(year: int, month: int) -> Path:
    """Downloads one file of raw data from the website of the TLC Trip Record Data and saves it to the data/raw folder.
    Args:
        year (int): The year of the data to download.
        month (int): The month of the data to download.
    Returns:
        Path: The path to the downloaded file.
    """
    # create the url
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
    response = requests.get(url)

    if response.status_code == 200:
        # create the path to the file
        #path = f'../data/raw/rides_{year}-{month:02d}.parquet' -> old version
        path = RAW_DATA_DIR / f'rides_{year}-{month:02d}.parquet'
        open(path, 'wb').write(response.content)
        return path #return Path(path) 
    else:
        print(f"Could not download file {url}")
        return None

# validate the raw data
def validate_raw_data(
        rides: pd.DataFrame,
        year: int,
        month: int,
) -> pd.DataFrame:
    """
    Validates the raw data and removes pickup_datetimes outside their valid range.
    Args:
        rides (pd.DataFrame): The raw data.
        year (int): The year of the data to validate.
        month (int): The month of the data to validate.
    """
    # keep only rides this month
    this_month_start = f'{year}-{month:02d}-01'
    next_month_start = f'{year}-{month+1:02d}-01' if month < 12 else f'{year+1}-01-01'
    rides = rides[(rides.pickup_datetime >= this_month_start)] #& (rides.pickup_datetime < next_month_start)]
    rides = rides[(rides.pickup_datetime < next_month_start)]
    return rides

# processing several files of raw data at once and load and validate the raw data
def load_raw_data(
        year: int,
        months: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Loads the raw data from the data/raw folder.
    Args:
        year (int): The year of the data to load.
        months (list): The months of the data to load.
    Returns:
        pd.DataFrame: The raw data.
    """
    rides = pd.DataFrame()
    
    # loop over months
    if months is None:
        # download data only for the months specified in the months argument
        months = range(1, 13)
    elif isinstance(months, int):
        # if months is an integer, convert it to a list
        # download data for the entire year (all 12 months)
        months = [months]

    for month in months:
        # chaching the file
        # download the file
        local_file = RAW_DATA_DIR / f'rides_{year}-{month:02d}.parquet'
        if not local_file.exists():
            try:
                # download the file from the NYC website
                print(f"Downloading file {year}-{month:02d}")
                download_one_file_of_raw_data(year, month)
            except:
                print(f"Could not download file {year}-{month:02d}")
                continue
        else:
            print(f"File {year}-{month:02d} already exists")

        # load the file into Pandas
        rides_one_month = pd.read_parquet(local_file)

        # keep only the columns we need
        rides_one_month = rides_one_month[['tpep_pickup_datetime', 'PULocationID']]
        
        # rename the columns
        rides_one_month = rides_one_month.rename(columns={
            'tpep_pickup_datetime': 'pickup_datetime',
            'PULocationID': 'pickup_location_id',
        })

        # validate the data
        rides_one_month = validate_raw_data(rides_one_month, year, month)

        # concatenate the data
        rides = pd.concat([rides, rides_one_month])#, axis=0, ignore_index=True)

        # # keep only time and origin of the ride
        # rides = rides[['pickup_datetime', 'pickup_location_id']]

    if rides.empty:
        print(f"No data for year {year} and months {months}")
        # return empty DataFrame
        return pd.DataFrame()
    else:
        # keep only time and origin of the ride
        rides = rides[['pickup_datetime', 'pickup_location_id']]
        return rides


## Option 2
# add extra column when were no rides option 2
from tqdm import tqdm

def add_missing_slots(ts_data: pd.DataFrame) -> pd.DataFrame:
    """
    Add necessary rows to the input 'ts_data' to make sure the output
    has a complete list of
    - pickup_hours
    - pickup_location_ids
    """

    # get all possible combinations of hour and location id
    #location_ids = ts_data['pickup_location_id'].unique()
    location_ids = range(1,ts_data['pickup_location_id'].max()+1)
    
    full_range = pd.date_range(
        ts_data['pickup_hour'].min(),
        ts_data['pickup_hour'].max(), 
        freq='H')
    output = pd.DataFrame()
    for location_id in tqdm(location_ids):

        # keep only rides for this location id
        rides_for_location = ts_data.loc[ts_data.pickup_location_id == location_id,['pickup_hour','rides']]

        if rides_for_location.empty:
            # add a dummy entry with 0 rides
            rides_for_location = pd.DataFrame.from_dict([
                {'pickup_hour': ts_data['pickup_hour'].max(), 'rides': 0}
            ])

        # add missing dates with 0 in a Series
        rides_for_location.set_index('pickup_hour', inplace=True)
        rides_for_location.index = pd.DatetimeIndex(rides_for_location.index)
        rides_for_location = rides_for_location.reindex(full_range, fill_value=0)

        # add back location id
        rides_for_location['pickup_location_id'] = location_id

        # add to output
        output = pd.concat([output, rides_for_location])

    # move the purchase date from the index to a dataframe column
    output = output.reset_index().rename(columns={'index': 'pickup_hour'})

    return output

## End Option 2


# transform the raw data into ts data
def transform_raw_data_into_ts_data(
        rides: pd.DataFrame,
) -> pd.DataFrame:
    """
    Transforms the raw data into time series data.
    Args:
        rides (pd.DataFrame): The raw data.
    Returns:
        pd.DataFrame: The time series data.
    """

    # sum rides per location and pickup_hour
    rides['pickup_hour'] = rides['pickup_datetime'].dt.floor('H')
    agg_rides = rides.groupby(['pickup_location_id', 'pickup_hour']).size().reset_index(name='rides')
    agg_rides = agg_rides.rename(columns={0: 'rides'}, inplace=True)

    # add rows for (locations, pickup_hours) with 0 rides
    agg_rides_all_slots = add_missing_slots(agg_rides)

    return agg_rides_all_slots














#### EXPERIMENTS ####
# ## Option 1 -> not working check later
# # add missing slots
# def add_misssing_slots(
#         rides: pd.DataFrame,
# ) -> pd.DataFrame:
#     """
#     Adds missing slots to the time series data.
#     Args:
#         rides (pd.DataFrame): The time series data.
#     Returns:
#         pd.DataFrame: The time series data with missing slots.
#     """
#     # get all locations and pickup_hours
#     # add extra column when were no rides option 1
#     rides = rides.set_index(['pickup_hour', 
#                              'pickup_location_id'])\
#                                 .unstack(fill_value=0).stack().reset_index()
#     return rides

## End Option 1
#### EXPERIMENTS ####