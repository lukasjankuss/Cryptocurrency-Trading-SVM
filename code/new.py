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
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

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
    'estimator__kernel': ['linear', 'rbf', 'poly', 'sigmoid']
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

# Get the complete cross-validation results for SVR
cv_results_svr = svr_grid_search.cv_results_
cv_results_svr_df = pd.DataFrame(cv_results_svr)
cv_results_svr_df_sorted = cv_results_svr_df.sort_values(by='mean_test_score', ascending=False)

print("Complete SVR cross-validation results: ")
print(cv_results_svr_df_sorted)

# Following is the code to compare different kernels
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
average_scores = []

for kernel in kernels:
    kernel_results = cv_results_svr_df[cv_results_svr_df['param_estimator__kernel'] == kernel]
    average_score = kernel_results['mean_test_score'].mean()
    average_scores.append([kernel, average_score])

average_scores_df = pd.DataFrame(average_scores, columns=['Kernel', 'Average Mean Test Score'])
print(average_scores_df)

# Fit both models to the training data
svr_regressor.fit(X_train, y_train)
rf_regressor.fit(X_train, y_train)

# Make predictions using both models
y_pred_svr = svr_regressor.predict(X_test)
y_pred_rf = rf_regressor.predict(X_test)

# Combine the predictions using a simple average
y_pred_ensemble = (y_pred_svr + y_pred_rf) / 2

# Evaluate the performance of the ensemble model and individual models
mse_ensemble = mean_squared_error(y_test, y_pred_ensemble)
mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
print('Ensemble Mean Squared Error:', mse_ensemble)
print('Ensemble Mean Absolute Error:', mae_ensemble)

mse_svr = mean_squared_error(y_test, y_pred_svr)
mae_svr = mean_absolute_error(y_test, y_pred_svr)
print('SVR Mean Squared Error:', mse_svr)
print('SVR Mean Absolute Error:', mae_svr)

mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print('Random Forest Mean Squared Error:', mse_rf)
print('Random Forest Mean Absolute Error:', mae_rf)

# Perform price trend prediction analysis
price_trends_pred = np.sign(y_pred_svr[:, 3] - y_pred_svr[:, 0])  # Predicted price trends (1: increase, -1: decrease)
price_trends_actual = np.sign(y_test[:, 3] - y_test[:, 0])  # Actual price trends from the test set

# Calculate additional evaluation metrics for price trend prediction
accuracy = np.mean(price_trends_pred == price_trends_actual)
precision = np.mean(price_trends_pred[price_trends_actual == 1] == 1)
recall = np.mean(price_trends_pred[price_trends_pred == 1] == 1)
f1_score = 2 * (precision * recall) / (precision + recall)

print('Price Trend Prediction Accuracy:', accuracy)
print('Price Trend Prediction Precision:', precision)
print('Price Trend Prediction Recall:', recall)
print('Price Trend Prediction F1-score:', f1_score)

# Implement and evaluate a simple trading strategy based on the predicted price trends
df_test = pd.DataFrame(y_test, columns=['Open', 'High', 'Low', 'Close'])
df_test['Predicted_Trend'] = price_trends_pred
df_test['Actual_Trend'] = price_trends_actual

# Define trading rules
df_test['Signal'] = np.where(df_test['Predicted_Trend'] == 1, 1, -1)
df_test['Returns'] = df_test['Signal'] * df_test['Close'].pct_change()
df_test['Cumulative_Returns'] = (1 + df_test['Returns']).cumprod()
df_test['Benchmark_Returns'] = df_test['Close'].pct_change().cumsum() + 1

# Evaluate trading strategy performance
strategy_returns = df_test['Cumulative_Returns'].iloc[-1]
benchmark_returns = df_test['Benchmark_Returns'].iloc[-1]

print('Trading Strategy Returns:', strategy_returns)
print('Benchmark Returns:', benchmark_returns)

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