from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.paths import RAW_DATA_DIR, TRANSFORMED_DATA_DIR

def download_one_file_of_raw_data(year: int, month: int) -> Path:
    """
    Downloads Parquet file with historical taxi rides for the given `year` and
    `month`
    """
    URL = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    response = requests.get(URL)

    if response.status_code == 200:
        path = RAW_DATA_DIR / f'rides_{year}-{month:02d}.parquet'
        open(path, "wb").write(response.content)
        return path
    else:
        raise Exception(f'{URL} is not available')


def validate_raw_data(
    rides: pd.DataFrame,
    year: int,
    month: int,
) -> pd.DataFrame:
    """
    Removes rows with pickup_datetimes outside their valid range
    """
    # keep only rides for this month
    this_month_start = f'{year}-{month:02d}-01'
    next_month_start = f'{year}-{month+1:02d}-01' if month < 12 else f'{year+1}-01-01'
    rides = rides[rides.pickup_datetime >= this_month_start]
    rides = rides[rides.pickup_datetime < next_month_start]
    
    return rides


def load_raw_data(
    year: int,
    months: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Loads raw data from local storage or downloads it from the NYC website
    """  
    rides = pd.DataFrame()
    
    if months is None:
        # download data only for the months specified by `months`
        months = list(range(1, 13))
    elif isinstance(months, int):
        # download data for the entire year (all months)
        months = [months]

    for month in months:
        
        local_file = RAW_DATA_DIR / f'rides_{year}-{month:02d}.parquet'
        if not local_file.exists():
            try:
                # download the file from the NYC website
                print(f'Downloading file {year}-{month:02d}')
                download_one_file_of_raw_data(year, month)
            except:
                print(f'{year}-{month:02d} file is not available')
                continue
        else:
            print(f'File {year}-{month:02d} was already in local storage') 

        # load the file into Pandas
        rides_one_month = pd.read_parquet(local_file)

        # rename columns
        rides_one_month = rides_one_month[['tpep_pickup_datetime', 'PULocationID']]
        rides_one_month.rename(columns={
            'tpep_pickup_datetime': 'pickup_datetime',
            'PULocationID': 'pickup_location_id',
        }, inplace=True)

        # validate the file
        rides_one_month = validate_raw_data(rides_one_month, year, month)

        # append to existing data
        rides = pd.concat([rides, rides_one_month])

    if rides.empty:
        # no data, so we return an empty dataframe
        return pd.DataFrame()
    else:
        # keep only time and origin of the ride
        rides = rides[['pickup_datetime', 'pickup_location_id']]
        return rides


def add_missing_slots(ts_data: pd.DataFrame) -> pd.DataFrame:
    """
    Add necessary rows to the input 'ts_data' to make sure the output
    has a complete list of
    - pickup_hours
    - pickup_location_ids
    """
    location_ids = range(1, ts_data['pickup_location_id'].max() + 1)

    full_range = pd.date_range(ts_data['pickup_hour'].min(),
                               ts_data['pickup_hour'].max(),
                               freq='H')
    output = pd.DataFrame()
    for location_id in tqdm(location_ids):

        # keep only rides for this 'location_id'
        ts_data_i = ts_data.loc[ts_data.pickup_location_id == location_id, ['pickup_hour', 'rides']]
        
        if ts_data_i.empty:
            # add a dummy entry with a 0
            ts_data_i = pd.DataFrame.from_dict([
                {'pickup_hour': ts_data['pickup_hour'].max(), 'rides': 0}
            ])

        # quick way to add missing dates with 0 in a Series
        # taken from https://stackoverflow.com/a/19324591
        ts_data_i.set_index('pickup_hour', inplace=True)
        ts_data_i.index = pd.DatetimeIndex(ts_data_i.index)
        ts_data_i = ts_data_i.reindex(full_range, fill_value=0)
        
        # add back `location_id` columns
        ts_data_i['pickup_location_id'] = location_id

        output = pd.concat([output, ts_data_i])
    
    # move the purchase_day from the index to a dataframe column
    output = output.reset_index().rename(columns={'index': 'pickup_hour'})
    
    return output


def transform_raw_data_into_ts_data(
    rides: pd.DataFrame
) -> pd.DataFrame:
    """"""
    # sum rides per location and pickup_hour
    rides['pickup_hour'] = rides['pickup_datetime'].dt.floor('H')
    agg_rides = rides.groupby(['pickup_hour', 'pickup_location_id']).size().reset_index()
    agg_rides.rename(columns={0: 'rides'}, inplace=True)

    # add rows for (locations, pickup_hours)s with 0 rides
    agg_rides_all_slots = add_missing_slots(agg_rides)

    return agg_rides_all_slots

########### Option 1 - w\o Parallelization ##########
def transform_ts_data_into_features_and_target(
    ts_data: pd.DataFrame,
    input_seq_len: int,
    step_size: int
) -> pd.DataFrame:
    """
    Slices and transposes data from time-series format into a (features, target)
    format that we can use to train Supervised ML models
    """
    assert set(ts_data.columns) == {'pickup_hour', 'rides', 'pickup_location_id'}

    location_ids = ts_data['pickup_location_id'].unique()
    features = pd.DataFrame()
    targets = pd.DataFrame()
    
    for location_id in tqdm(location_ids):
        
        # keep only ts data for this `location_id`
        ts_data_one_location = ts_data.loc[
            ts_data.pickup_location_id == location_id, 
            ['pickup_hour', 'rides']
        ].sort_values(by=['pickup_hour'])

        # pre-compute cutoff indices to split dataframe rows
        indices = get_cutoff_indices_features_and_target(
            ts_data_one_location,
            input_seq_len,
            step_size
        )

        # slice and transpose data into numpy arrays for features and targets
        n_examples = len(indices)
        x = np.ndarray(shape=(n_examples, input_seq_len), dtype=np.float32)
        y = np.ndarray(shape=(n_examples), dtype=np.float32)
        pickup_hours = []
        for i, idx in enumerate(indices):
            x[i, :] = ts_data_one_location.iloc[idx[0]:idx[1]]['rides'].values
            y[i] = ts_data_one_location.iloc[idx[1]:idx[2]]['rides'].values
            pickup_hours.append(ts_data_one_location.iloc[idx[1]]['pickup_hour'])

        # numpy -> pandas
        features_one_location = pd.DataFrame(
            x,
            columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(input_seq_len))]
        )
        features_one_location['pickup_hour'] = pickup_hours
        features_one_location['pickup_location_id'] = location_id

        # numpy -> pandas
        targets_one_location = pd.DataFrame(y, columns=[f'target_rides_next_hour'])

        # concatenate results
        features = pd.concat([features, features_one_location])
        targets = pd.concat([targets, targets_one_location])

    features.reset_index(inplace=True, drop=True)
    targets.reset_index(inplace=True, drop=True)

    return features, targets['target_rides_next_hour']
########### Option 1 - w\o Parallelization ##########


# ########### Option 2 - with Parallelization ##########
# def transform_ts_data_into_features_and_targets(
#         ts_data: pd.DataFrame,
#         input_seq_len: int,
#         step_size: int,
# ) -> pd.DataFrame:
#     """
#     This function transforms the time series data into features and targets.
#     Slices and transposes data from time series to supervised learning problem(features and targets).
#     """
#     assert set(ts_data.columns) == {
#         'pickup_location_id',
#         'pickup_hour', 
#         'rides'},'The columns of the dataframe are not correct.'

#     # init
#     location_ids = ts_data.pickup_location_id.unique()
#     features = []
#     targets = []

#     # loop over the locations in parallel
#     results = Parallel(n_jobs=-1)(delayed(transform_one_location)(
#         ts_data.loc[ts_data.pickup_location_id == location_id,
#                     ['pickup_hour', 'rides']], input_seq_len, step_size)
#         for location_id in tqdm(location_ids))

#     # extract the results
#     for ts_data_one_location, targets_ts_data_one_location in results:
#         features.append(ts_data_one_location)
#         targets.append(targets_ts_data_one_location)

#     # concat the results from the previous iterations
#     features = pd.concat(features, axis=0, ignore_index=True)
#     targets = pd.concat(targets, axis=0, ignore_index=True)

#     return features, targets

# def transform_one_location(ts_data_one_location, input_seq_len, step_size):
#     # pre-compute cutoff indices to split dataframe rows
#     indices = get_cutoff_indices_features_and_target(ts_data_one_location, input_seq_len, step_size)

#     # slide and transpose the data into numpy arrays for features and targets
#     n_examples = len(indices)
#     X = np.ndarray(shape=(n_examples, input_seq_len), dtype=np.float32)
#     y = np.ndarray(shape=(n_examples), dtype=np.float32)
#     pickup_hours = []

#     # loop over the indices to generate the values for features and targets
#     for i, idx in enumerate(indices):
#         # get the features
#         # extracting the first and the second index of the cutoff indices
#         X[i, :] = ts_data_one_location.iloc[idx[0]:idx[1]]['rides'].values

#         # get the target
#         # extracting the second and the third index of the cutoff indices (mid and right side ones)
#         y[i] = ts_data_one_location.iloc[idx[1]:idx[2]]['rides'].values

#         # store the pickup hours
#         pickup_hours.append(ts_data_one_location.iloc[idx[1]]['pickup_hour'])

#     # numpy -> pandas for X
#     # convert numpy arrays to pandas dataframes for X values
#     ts_data_one_location = pd.DataFrame(
#         X, 
#         columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(input_seq_len))])
    
#     # numpy -> pandas for y
#     # transform the target into a dataframe
#     targets_ts_data_one_location = pd.DataFrame(y, columns=[f'target_rides_next_hour'])
    
#     return ts_data_one_location, targets_ts_data_one_location['target_rides_next_hour']

# ########### Option 2 - with Parallelization ##########



def get_cutoff_indices_features_and_target(
    data: pd.DataFrame,
    input_seq_len: int,
    step_size: int
    ) -> list:
    """
    Returns a list of tuples of indices that can be used to slice a dataframe
    into (features, target) pairs.
    """

    stop_position = len(data) - 1
    
    # Start the first sub-sequence at index position 0
    subseq_first_idx = 0
    subseq_mid_idx = input_seq_len
    subseq_last_idx = input_seq_len + 1
    indices = []
    
    while subseq_last_idx <= stop_position:
        indices.append((subseq_first_idx, subseq_mid_idx, subseq_last_idx))
        subseq_first_idx += step_size
        subseq_mid_idx += step_size
        subseq_last_idx += step_size

    return indices











########################################################################################
# ## MY APPROACH STARTS HERE ## -> investigate the file later or delete it
# # data path and import 
# from pathlib import Path
# from datetime import datetime, timedelta
# from typing import List, Optional
# import tqdm

# import pandas as pd
# import numpy as np
# import requests

# from src.paths import RAW_DATA_DIR, TRANSFORMED_DATA_DIR, PARENT_DIR, DATA_DIR

# # download one file of raw data
# def download_one_file_of_raw_data(year: int, month: int) -> Path:
#     """Downloads one file of raw data from the website of the TLC Trip Record Data and saves it to the data/raw folder.
#     Args:
#         year (int): The year of the data to download.
#         month (int): The month of the data to download.
#     Returns:
#         Path: The path to the downloaded file.
#     """
#     # create the url
#     url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
#     response = requests.get(url)

#     if response.status_code == 200:
#         # create the path to the file
#         #path = f'../data/raw/rides_{year}-{month:02d}.parquet' -> old version
#         path = RAW_DATA_DIR / f'rides_{year}-{month:02d}.parquet'
#         open(path, 'wb').write(response.content)
#         return path #return Path(path) 
#     else:
#         print(f"Could not download file {url}")
#         return None

# # validate the raw data
# def validate_raw_data(
#         rides: pd.DataFrame,
#         year: int,
#         month: int,
# ) -> pd.DataFrame:
#     """
#     Validates the raw data and removes pickup_datetimes outside their valid range.
#     Args:
#         rides (pd.DataFrame): The raw data.
#         year (int): The year of the data to validate.
#         month (int): The month of the data to validate.
#     """
#     # keep only rides this month
#     this_month_start = f'{year}-{month:02d}-01'
#     next_month_start = f'{year}-{month+1:02d}-01' if month < 12 else f'{year+1}-01-01'
#     rides = rides[(rides.pickup_datetime >= this_month_start)] #& (rides.pickup_datetime < next_month_start)]
#     rides = rides[(rides.pickup_datetime < next_month_start)]
#     return rides

# # processing several files of raw data at once and load and validate the raw data
# def load_raw_data(
#         year: int,
#         months: Optional[List[int]] = None,
# ) -> pd.DataFrame:
#     """
#     Loads the raw data from the data/raw folder.
#     Args:
#         year (int): The year of the data to load.
#         months (list): The months of the data to load.
#     Returns:
#         pd.DataFrame: The raw data.
#     """
#     rides = pd.DataFrame()
    
#     # loop over months
#     if months is None:
#         # download data only for the months specified in the months argument
#         months = range(1, 13)
#     elif isinstance(months, int):
#         # if months is an integer, convert it to a list
#         # download data for the entire year (all 12 months)
#         months = [months]

#     for month in months:
#         # chaching the file
#         # download the file
#         local_file = RAW_DATA_DIR / f'rides_{year}-{month:02d}.parquet'
#         if not local_file.exists():
#             try:
#                 # download the file from the NYC website
#                 print(f"Downloading file {year}-{month:02d}")
#                 download_one_file_of_raw_data(year, month)
#             except:
#                 print(f"Could not download file {year}-{month:02d}")
#                 continue
#         else:
#             print(f"File {year}-{month:02d} already exists")

#         # load the file into Pandas
#         rides_one_month = pd.read_parquet(local_file)

#         # keep only the columns we need
#         rides_one_month = rides_one_month[['tpep_pickup_datetime', 'PULocationID']]
        
#         # rename the columns
#         rides_one_month = rides_one_month.rename(columns={
#             'tpep_pickup_datetime': 'pickup_datetime',
#             'PULocationID': 'pickup_location_id',
#         })

#         # validate the data
#         rides_one_month = validate_raw_data(rides_one_month, year, month)

#         # concatenate the data
#         rides = pd.concat([rides, rides_one_month])#, axis=0, ignore_index=True)

#         # # keep only time and origin of the ride
#         # rides = rides[['pickup_datetime', 'pickup_location_id']]

#     if rides.empty:
#         print(f"No data for year {year} and months {months}")
#         # return empty DataFrame
#         return pd.DataFrame()
#     else:
#         # keep only time and origin of the ride
#         rides = rides[['pickup_datetime', 'pickup_location_id']]
#         return rides


# ## Option 2
# # add extra column when were no rides option 2
# from tqdm import tqdm

# def add_missing_slots(ts_data: pd.DataFrame) -> pd.DataFrame:
#     """
#     Add necessary rows to the input 'ts_data' to make sure the output
#     has a complete list of
#     - pickup_hours
#     - pickup_location_ids
#     """

#     # get all possible combinations of hour and location id
#     #location_ids = ts_data['pickup_location_id'].unique()
#     location_ids = range(1,ts_data['pickup_location_id'].max()+1)
    
#     full_range = pd.date_range(
#         ts_data['pickup_hour'].min(),
#         ts_data['pickup_hour'].max(), 
#         freq='H')
#     output = pd.DataFrame()
#     for location_id in tqdm(location_ids):

#         # keep only rides for this location id
#         rides_for_location = ts_data.loc[ts_data.pickup_location_id == location_id,['pickup_hour','rides']]

#         if rides_for_location.empty:
#             # add a dummy entry with 0 rides
#             rides_for_location = pd.DataFrame.from_dict([
#                 {'pickup_hour': ts_data['pickup_hour'].max(), 'rides': 0}
#             ])

#         # add missing dates with 0 in a Series
#         rides_for_location.set_index('pickup_hour', inplace=True)
#         rides_for_location.index = pd.DatetimeIndex(rides_for_location.index)
#         rides_for_location = rides_for_location.reindex(full_range, fill_value=0)

#         # add back location id
#         rides_for_location['pickup_location_id'] = location_id

#         # add to output
#         output = pd.concat([output, rides_for_location])

#     # move the purchase date from the index to a dataframe column
#     output = output.reset_index().rename(columns={'index': 'pickup_hour'})

#     return output

# ## End Option 2


# # transform the raw data into ts data
# def transform_raw_data_into_ts_data(
#         rides: pd.DataFrame,
# ) -> pd.DataFrame:
#     """
#     Transforms the raw data into time series data.
#     Args:
#         rides (pd.DataFrame): The raw data.
#     Returns:
#         pd.DataFrame: The time series data.
#     """

#     # sum rides per location and pickup_hour
#     rides['pickup_hour'] = rides['pickup_datetime'].dt.floor('H')
#     agg_rides = rides.groupby(['pickup_location_id', 'pickup_hour']).size().reset_index(name='rides')
#     agg_rides = agg_rides.rename(columns={0: 'rides'}, inplace=True)

#     # add rows for (locations, pickup_hours) with 0 rides
#     agg_rides_all_slots = add_missing_slots(agg_rides)

#     return agg_rides_all_slots

########################################################################################



################################ EXPERIMENTS ###########################################
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
################################ EXPERIMENTS ###########################################