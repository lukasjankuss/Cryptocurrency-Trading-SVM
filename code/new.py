# Import libraries and modules
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

# Load the dataset
df = pd.read_csv('data/BTC-USDv2.csv')
df.dropna(inplace=True)

# Convert the date column to Unix timestamps
df['Date'] = df['Date'].apply(lambda x: int(datetime.strptime(x, '%d/%m/%Y').timestamp()))

# Feature Engineering
df['Price Change'] = df['Close'].pct_change()
df['Volume Change'] = df['Volume'].pct_change()

# Add a rolling average
df['Rolling_Avg'] = df['Close'].rolling(window=5).mean()

df = df.dropna()

# Prepare the data for modeling
X = df[['Date', 'Volume', 'Price Change', 'Volume Change', 'Rolling_Avg']].values
y = df[['Open', 'High', 'Low', 'Close']].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the models
svr = SVR()
rf = RandomForestRegressor(random_state=0)

# Create the MultiOutputRegressor wrappers
svr_regressor = MultiOutputRegressor(svr)
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

# Make predictions using both models
y_pred_svr = svr_regressor.predict(X_test)
y_pred_rf = rf_regressor.predict(X_test)

# Combine the predictions using a simple average
y_pred_ensemble = (y_pred_svr + y_pred_rf) / 2

# Evaluate the performance of the ensemble model
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred_ensemble)
print('Mean Squared Error:', mse)

# Plot the actual vs predicted values
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
axs = axs.ravel()

for i, label in enumerate(['Open', 'High', 'Low', 'Close']):
    sns.lineplot(data=pd.DataFrame({'Actual': y_test[:, i], 'Predicted': y_pred_ensemble[:, i]}), ax=axs[i])
    axs[i].set(title=f'Actual vs Predicted {label} Prices', xlabel='Sample', ylabel='Price')
    axs[i].legend()

plt.show()

# Prepare future dates
future_dates = pd.date_range(start='2023-03-21', end='2023-04-20', freq='D')  # 30 days

# Assume that the 'Volume', 'Price Change', 'Volume Change', 'Rolling_Avg' remain the same as the last date in your original data
future_X = np.tile(X[-1], (len(future_dates), 1))
future_X[:, 0] = [int(x.timestamp()) for x in future_dates]  # Replace 'Date' with future dates in Unix timestamps

# Scale the future_X using the same scaler object
future_X_scaled = scaler.transform(future_X)

# Use the ensemble model to make predictions on the scaled feature matrix
y_future_pred_svr = svr_regressor.predict(future_X_scaled)
y_future_pred_rf = rf_regressor.predict(future_X_scaled)
y_future_pred_ensemble = (y_future_pred_svr + y_future_pred_rf) / 2

# Print the predicted Open, High, Low, and Close prices for the future dates
future_prices_df = pd.DataFrame(y_future_pred_ensemble, columns=['Open', 'High', 'Low', 'Close'])
future_prices_df.index = future_dates
print(future_prices_df)