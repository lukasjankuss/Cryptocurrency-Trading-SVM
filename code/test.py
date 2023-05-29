# Load the necessary libraries
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# ... rest of your code up to the model creation ...

# Create the SVR model
svr = SVR()

# Create the Random Forest Regressor model
rf = RandomForestRegressor(random_state=0)

# Create the MultiOutputRegressor with the SVR model
svr_regressor = MultiOutputRegressor(svr)

# Create the MultiOutputRegressor with the Random Forest Regressor model
rf_regressor = MultiOutputRegressor(rf)

# Define the parameter grid for SVR
svr_param_grid = {
    'estimator__C': [0.1, 1, 10, 100],
    'estimator__epsilon': [0.01, 0.1, 1],
    'estimator__kernel': ['linear', 'rbf']
}

# Define the parameter grid for Random Forest
rf_param_grid = {
    'estimator__n_estimators': [10, 50, 100, 200],
    'estimator__max_depth': [None, 10, 50, 100],
    'estimator__min_samples_split': [2, 5, 10]
}

# Create GridSearchCV objects for SVR and Random Forest
svr_grid_search = GridSearchCV(svr_regressor, svr_param_grid, cv=5, n_jobs=-1, verbose=2)
rf_grid_search = GridSearchCV(rf_regressor, rf_param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the grid search objects
svr_grid_search.fit(X_train, y_train)
rf_grid_search.fit(X_train, y_train)

# Get the best parameters
best_svr_params = svr_grid_search.best_params_
best_rf_params = rf_grid_search.best_params_

print("Best SVR parameters: ", best_svr_params)
print("Best Random Forest parameters: ", best_rf_params)

# Create the SVR model with the best parameters
best_svr = SVR(kernel=best_svr_params['estimator__kernel'], C=best_svr_params['estimator__C'], epsilon=best_svr_params['estimator__epsilon'])

# Create the Random Forest Regressor model with the best parameters
best_rf = RandomForestRegressor(n_estimators=best_rf_params['estimator__n_estimators'], max_depth=best_rf_params['estimator__max_depth'], min_samples_split=best_rf_params['estimator__min_samples_split'], random_state=0)

# Create the MultiOutputRegressor with the best SVR model
svr_regressor = MultiOutputRegressor(best_svr)

# Create the MultiOutputRegressor with the best Random Forest Regressor model
rf_regressor = MultiOutputRegressor(best_rf)

# Fit both models to the training data
svr_regressor.fit(X_train, y_train)
rf_regressor.fit(X_train, y_train)

# ... rest of your code ...

