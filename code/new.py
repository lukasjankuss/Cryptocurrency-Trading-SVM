import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('data/Coinmetrics.csv')
df.dropna(inplace=True)

# Convert the date column to Unix timestamps
df['Date'] = df['Date'].apply(lambda x: int(datetime.strptime(x, '%Y-%m-%d').timestamp()))

# Prepare the data for modeling
X = df['Date'].values.reshape(-1, 1)
y = df['Price'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# Scale the features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM model
regressor = SVR(kernel='rbf', C=10, epsilon=0.1)
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)

# Evaluate the performance of the model
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# Plot the actual vs predicted values
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}), ax=ax)
ax.set(title='Actual vs Predicted Values', xlabel='Sample', ylabel='Price')

# Add a shaded region for the confidence interval
ci = 1.96 * np.std(y_pred) / np.mean(y_pred)
ax.fill_between(x=range(len(y_pred)), y1=y_pred - ci, y2=y_pred + ci, alpha=0.2)

# Add a horizontal line at the mean value
ax.axhline(np.mean(y_test), ls='--', color='gray', label='Mean')

# Add a legend
ax.legend()

# Add a title to the figure
fig.suptitle('Support Vector Regression on Bitcoin Price', fontsize=16)

plt.show()

