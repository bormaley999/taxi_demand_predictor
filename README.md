# Taxi Demand Predictor Service
This repo is subject to change:
- Add new cities;
- Add some visuals;
- Other funcy or not so fancy stuff;


## Data Source
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

## Notes Section
- I've used Python Poetry (`.toml` file) to keep all the dependencies clear and concise. 
- Use this slackthread if encountered problems with Poetry: https://stackoverflow.com/questions/70003829/poetry-installed-but-poetry-command-not-found
- Used Optuna for Hyperparameter Tuning
- For MacOS (Monterey 12.6) users you'll likely have to fight with LightGBM install (stackoverflow will help you)
- smth


## To Do's
- [x]Use Feature Importance for 8 weeks of data - XGBoost;
- [x]Use Feature Importance for 4 weeks of data - LighGBM;
- [x]Use Feature Importance for 8 weeks of data - LighGBM;
- Visualize Feature Importance via SHAP(need to fix Poetry dependencies);
- Test Exponential Smoothing;
- Test Prophet;
- Test ARIMA;
- Test SARIMA;
- Test Weights and Biases tool
- Test MLFlow for MLOPS tasks
- Test DataIKU for MLOPS tasks



## Done (extra to the original task)
- Added a simple Feature Importance job for `07_xboost_model.ipynb` notebook;
- Used 8 weeks dataset for training as an extension of the previous 4 weeks dataset;
- Used my feature importance apporach, to check with the original approach used by Paulo