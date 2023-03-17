import numpy as np
import pandas as pd
from sklearn.svm import SVR
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
df['Date'] = df['Date'].apply(lambda x: int(datetime.strptime(x, '%Y-%m-%d').timestamp()))

# Prepare the data for modeling
X = df['Date'].values.reshape(-1, 1)
y = df[['Open', 'High', 'Low', 'Close']].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the parameter grid for the GridSearchCV
param_grid = {
    'estimator__kernel': ['rbf', 'linear', 'poly'],
    'estimator__C': [0.1, 1, 10, 100],
    'estimator__epsilon': [0.01, 0.1, 1],
}

# Create the SVR model
svr = SVR()

# Create the MultiOutputRegressor with the SVR model
regressor = MultiOutputRegressor(svr)

# Create the GridSearchCV object
grid_search = GridSearchCV(regressor, param_grid, cv=5, verbose=2, n_jobs=-1)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters
print("Best parameters found: ", grid_search.best_params_)

# Make predictions using the best model
best_regressor = grid_search.best_estimator_
y_pred = best_regressor.predict(X_test)

# Evaluate the performance of the best model
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# Plot the actual vs predicted values
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
axs = axs.ravel()

for i, label in enumerate(['Open', 'High', 'Low', 'Close']):
    sns.lineplot(data=pd.DataFrame({'Actual': y_test[:, i], 'Predicted': y_pred[:, i]}), ax=axs[i])
    axs[i].set(title=f'Actual vs Predicted {label} Prices', xlabel='Sample', ylabel='Price')
    axs[i].legend()

# Add a title to the figure
fig.suptitle('Support Vector Regression on Bitcoin Price Predictions', fontsize=16)

plt.show()
