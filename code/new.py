# Import libraries and modules
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('data/ETH-USD.csv')
df.dropna(inplace=True)

df['Date'] = df['Date'].apply(lambda x: int(datetime.strptime(x, '%d/%m/%Y').timestamp()))

df['Price Change'] = df['Close'].pct_change()
df['Volume Change'] = df['Volume'].pct_change()

df['Rolling_Avg'] = df['Close'].rolling(window=5).mean()

df = df.dropna()

X = df[['Date', 'Volume', 'Price Change', 'Volume Change', 'Rolling_Avg']].values
y = df[['Open', 'High', 'Low', 'Close']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svr = SVR()

svr_regressor = MultiOutputRegressor(svr)

svr_param_grid = {
    'estimator__C': [0.1, 1, 10, 100],
    'estimator__epsilon': [0.01, 0.1, 1],
    'estimator__kernel': ['linear', 'rbf', 'poly', 'sigmoid']
}

svr_grid_search = GridSearchCV(svr_regressor, svr_param_grid, cv=5, n_jobs=-1, verbose=2)
svr_grid_search.fit(X_train, y_train)

best_svr_params = svr_grid_search.best_params_

print("Best SVR parameters: ", best_svr_params)

best_svr = SVR(kernel=best_svr_params['estimator__kernel'], C=best_svr_params['estimator__C'], epsilon=best_svr_params['estimator__epsilon'])

svr_regressor = MultiOutputRegressor(best_svr)

cv_results_svr = svr_grid_search.cv_results_
cv_results_svr_df = pd.DataFrame(cv_results_svr)
cv_results_svr_df_sorted = cv_results_svr_df.sort_values(by='mean_test_score', ascending=False)

print("Complete SVR cross-validation results: ")
print(cv_results_svr_df_sorted)

kernels = ['linear', 'rbf', 'poly', 'sigmoid']
average_scores = []

for kernel in kernels:
    kernel_results = cv_results_svr_df[cv_results_svr_df['param_estimator__kernel'] == kernel]
    average_score = kernel_results['mean_test_score'].mean()
    average_scores.append([kernel, average_score])

average_scores_df = pd.DataFrame(average_scores, columns=['Kernel', 'Average Mean Test Score'])
print(average_scores_df)

svr_regressor.fit(X_train, y_train)

y_pred_svr = svr_regressor.predict(X_test)

# Initialize a list to store accuracy results for the SVR model
svr_accuracies = []

# Loop over different training periods
for period in range(100, len(X_train), 100):  # Start with a training set of size 100 and increase by 100 in each iteration
    # Train the SVR model
    svr_regressor.fit(X_train[:period], y_train[:period])

    # Make predictions using the SVR model
    y_pred_svr = svr_regressor.predict(X_test)

    # Calculate accuracy for the SVR model and append to the list
    svr_accuracy = mean_absolute_error(y_test, y_pred_svr)

    svr_accuracies.append(svr_accuracy)

# Plot accuracy over different training periods for the SVR model
plt.figure(figsize=(10, 5))
plt.plot(range(100, len(X_train), 100), svr_accuracies, label='SVR')
plt.xlabel('Training Period')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()

# Plot distribution of scores for each kernel
plt.figure(figsize=(10, 5))
for kernel in kernels:
    # Filter results for each kernel
    kernel_results = cv_results_svr_df[cv_results_svr_df['param_estimator__kernel'] == kernel]
    
    # Plot histogram of mean test scores
    plt.hist(kernel_results['mean_test_score'], bins=20, alpha=0.5, label=f'{kernel}')

plt.xlabel('Mean Test Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()

mse_svr = mean_squared_error(y_test, y_pred_svr)
mae_svr = mean_absolute_error(y_test, y_pred_svr)  # Calculate the mean absolute error for the SVR model
print('SVR Mean Squared Error:', mse_svr)  # Print the mean squared error for the SVR model
print('SVR Mean Absolute Error:', mae_svr)  # Print the mean absolute error for the SVR model

# Perform price trend prediction analysis
price_trends_pred = np.sign(y_pred_svr[:, 3] - y_pred_svr[:, 0])  # Predicted price trends (1: increase, -1: decrease)
price_trends_actual = np.sign(y_test[:, 3] - y_test[:, 0])  # Actual price trends from the test set

# Calculate additional evaluation metrics for price trend prediction
accuracy = np.mean(price_trends_pred == price_trends_actual)  # Calculate the accuracy of price trend prediction
precision = np.mean(price_trends_pred[price_trends_actual == 1] == 1)  # Calculate the precision of price trend prediction
recall = np.mean(price_trends_pred[price_trends_pred == 1] == 1)  # Calculate the recall of price trend prediction
f1_score = 2 * (precision * recall) / (precision + recall)  # Calculate the F1-score of price trend prediction

print('Price Trend Prediction Accuracy:', accuracy)  # Print the accuracy of price trend prediction
print('Price Trend Prediction Precision:', precision)  # Print the precision of price trend prediction
print('Price Trend Prediction Recall:', recall)  # Print the recall of price trend prediction
print('Price Trend Prediction F1-score:', f1_score)  # Print the F1-score of price trend prediction

# Implement and evaluate a simple trading strategy based on the predicted price trends
df_test = pd.DataFrame(y_test, columns=['Open', 'High', 'Low', 'Close'])  # Create a DataFrame for the test data
df_test['Predicted_Trend'] = price_trends_pred  # Add predicted price trends to the DataFrame
df_test['Actual_Trend'] = price_trends_actual  # Add actual price trends to the DataFrame

# Define trading rules
df_test['Signal'] = np.where(df_test['Predicted_Trend'] == 1, 1, -1)  # Generate trading signals based on predicted trends
df_test['Returns'] = df_test['Signal'] * df_test['Close'].pct_change()  # Calculate returns based on trading signals
df_test['Cumulative_Returns'] = (1 + df_test['Returns']).cumprod()  # Calculate cumulative returns
df_test['Benchmark_Returns'] = df_test['Close'].pct_change().cumsum() + 1  # Calculate benchmark returns

# Evaluate trading strategy performance
strategy_returns = df_test['Cumulative_Returns'].iloc[-1]  # Get the final cumulative returns of the trading strategy
benchmark_returns = df_test['Benchmark_Returns'].iloc[-1]  # Get the final benchmark returns

print('Trading Strategy Returns:', strategy_returns)  # Print the returns of the trading strategy
print('Benchmark Returns:', benchmark_returns)  # Print the benchmark returns

# Plot the actual vs predicted values
fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # Create subplots for each price variable
axs = axs.ravel()  # Flatten the subplots array

for i, label in enumerate(['Open', 'High', 'Low', 'Close']):
    sns.lineplot(data=pd.DataFrame({'Actual': y_test[:, i], 'Predicted': y_pred_svr[:, i]}), ax=axs[i])
    axs[i].set(title=f'Actual vs Predicted {label} Prices', xlabel='Sample', ylabel='Price')
    axs[i].legend()

plt.show()

# Prepare future dates
future_dates = pd.date_range(start='2023-03-21', end='2023-04-20', freq='D')  # Generate future dates (30 days)

# Assume that the 'Volume', 'Price Change', 'Volume Change', 'Rolling_Avg' remain the same as the last date in your original data
future_X = np.tile(X[-1], (len(future_dates), 1))  # Repeat the last row of X for the length of future dates
future_X[:, 0] = [int(x.timestamp()) for x in future_dates]  # Replace 'Date' with future dates in Unix timestamps

# Scale the future_X using the same scaler object
future_X_scaled = scaler.transform(future_X)  # Scale the future_X using the same scaler object

# Use the ensemble model to make predictions on the scaled feature matrix
y_future_pred_svr = svr_regressor.predict(future_X_scaled)  # Make predictions for future prices using the SVR model

# Print the predicted Open, High, Low, and Close prices for the future dates
future_prices_df = pd.DataFrame(y_future_pred_svr, columns=['Open', 'High', 'Low', 'Close'])
future_prices_df.index = future_dates
print(future_prices_df)
