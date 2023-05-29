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

# Create the SVR model with the best parameters found from the previous GridSearchCV
best_svr = SVR(kernel='rbf', C=100, epsilon=0.01)

# Create the Random Forest Regressor model
rf = RandomForestRegressor(n_estimators=100, random_state=0)

# Create the MultiOutputRegressor with the SVR model
svr_regressor = MultiOutputRegressor(best_svr)

# Create the MultiOutputRegressor with the Random Forest Regressor model
rf_regressor = MultiOutputRegressor(rf)

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

# Prepare future dates and corresponding volume data
future_dates = pd.date_range(start='2023-03-21', end='2023-04-20', freq='D')  # 30 days
np.random.seed(42)

future_volumes = np.random.choice(df['Volume'], len(future_dates))  # Randomly selecting historical volumes as a proxy

# Feature Engineering for future data
future_price_change = np.append([0], np.diff(future_volumes) / future_volumes[:-1])  
future_vol_change = np.append([0], np.diff(future_volumes) / future_volumes[:-1])

# Pad the price change and volume change arrays with an initial value to match lengths
future_price_change = np.pad(future_price_change, (1, 0), mode='edge')[1:]
future_vol_change = np.pad(future_vol_change, (1, 0), mode='edge')[1:]

# Placeholder for future rolling averages
future_rolling_avgs = np.full(len(future_dates), df['Rolling_Avg'].iloc[-1])

# Convert future dates to Unix timestamps and create a feature matrix
future_dates_unix = [int(x.timestamp()) for x in future_dates]
X_future = np.column_stack((future_dates_unix, future_volumes, future_price_change, future_vol_change, future_rolling_avgs))

# Scale the feature matrix using the same scaler object
X_future_scaled = scaler.transform(X_future)

# Use the ensemble model to make predictions on the scaled feature matrix
y_future_pred_svr = svr_regressor.predict(X_future_scaled)
y_future_pred_rf = rf_regressor.predict(X_future_scaled)
y_future_pred_ensemble = (y_future_pred_svr + y_future_pred_rf) / 2

# Print the predicted Open, High, Low, and Close prices for the future dates
future_prices_df = pd.DataFrame(y_future_pred_ensemble, columns=['Open', 'High', 'Low', 'Close'])
future_prices_df.index = future_dates
print(future_prices_df)

# Add a title to the figure
fig.suptitle('Bitcoin Price Predictions', fontsize=16)

plt.show()
