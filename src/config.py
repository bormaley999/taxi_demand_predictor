import os
from dotenv import load_dotenv
from src.paths import PARENT_DIR

# load keu-value pairs from .env file located in the parent directory
load_dotenv(PARENT_DIR / ".env")

HOPSWORKS_PROJECT_NAME = "taxi_demand_bormaley"
try:
    HOPSWORKS_API_KEY=os.environ['HOPSWORKS_API_KEY']
except:
    raise Exception("Create an .env file on the project root directory and add the HOPSWORKS_API_KEY variable with the value of your API key")

# define the feature group
FEATURE_GROUP_NAME = 'time_series_hourly_feature_group'
FEATURE_GROUP_VERSION = 1