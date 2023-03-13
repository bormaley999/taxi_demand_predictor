### this file contains all the imports and functions that are used in the notebooks
## libs import
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings
import lux
import tqdm
import requests
from typing import List, Optional
import plotly.express as px

## remove warnings
warnings.filterwarnings('ignore')

## retina display -> delete this line if you don't have a retina display
#%config InlineBackend.figure_format = 'retina'

## lux-api details increase the size of the visualizations
# lux.config.default_display = "lux-widget"
# lux.config.plotting_backend = "plotly"
# lux.config.default_display_size = "large"


