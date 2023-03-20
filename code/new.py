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

# Prepare the data for modeling
X = df[['Date', 'Volume']].values
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

# Add a title to the figure
fig.suptitle('Bitcoin Price Predictions', fontsize=16)

plt.show()
