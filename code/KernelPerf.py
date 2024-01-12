# Import libraries and modules
import numpy as np  # Import numpy for numeric computations
import pandas as pd  # Import pandas for data manipulation and analysis
from sklearn.svm import SVR  # Import Support Vector Regression model from scikit-learn
from sklearn.multioutput import MultiOutputRegressor  # Import MultiOutputRegressor for multiple target regression
from sklearn.model_selection import train_test_split, GridSearchCV  # Import function to split data and perform grid search
from sklearn.preprocessing import MinMaxScaler  # Import MinMaxScaler for normalization
from datetime import datetime  # Import datetime for date and time manipulation
import matplotlib.pyplot as plt  # Import matplotlib for data visualization
import seaborn as sns  # Import seaborn for data visualization
from sklearn.metrics import mean_absolute_error  # Import mean_absolute_error for model evaluation
from sklearn.metrics import mean_squared_error  # Import mean_squared_error for model evaluation
import time  # Import the time module
from sklearn.utils import shuffle # Import the shuffle module
from sklearn.model_selection import learning_curve  # Import learning_curve from scikit-learn

# Load the dataset
df = pd.read_csv('data/Ethereum/ETH-USD.csv')  # Load the CSV data
df.dropna(inplace=True)  # Remove any NA or missing values from the dataframe

# Convert the 'Date' column to UNIX timestamp
df['Date'] = df['Date'].apply(lambda x: int(datetime.strptime(x, '%d/%m/%Y').timestamp()))  # Convert 'Date' to UNIX timestamp

# Calculate the percentage change in 'Close' price and 'Volume'
df['Price Change'] = df['Close'].pct_change()  # Calculate percentage change in 'Close' price
df['Volume Change'] = df['Volume'].pct_change()  # Calculate percentage change in 'Volume'

# Calculate the rolling average of the past 5 'Close' prices
df['Rolling_Avg'] = df['Close'].rolling(window=5).mean()  # Calculate rolling average of past 5 'Close' prices

# Drop any remaining rows with NA or missing values
df = df.dropna()  # Drop any rows with missing values

# Define the features and target variables
X = df[['Date', 'Volume', 'Price Change', 'Volume Change', 'Rolling_Avg']].values  # Define features
y = df[['Open', 'High', 'Low', 'Close']].values  # Define target variables

# Split the data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  # Split the data

# Normalize the features to a range of [0, 1]
scaler = MinMaxScaler()  # Initialize the MinMaxScaler
X_train = scaler.fit_transform(X_train)  # Fit the scaler to the training data and transform it
X_test = scaler.transform(X_test)  # Transform the testing data with the scaler

# Define the base SVR model
svr = SVR()  # Initialize the SVR model

# Define the multioutput regressor with the base SVR model
svr_regressor = MultiOutputRegressor(svr)  # Wrap the SVR model to handle multiple outputs

# Define the hyperparameters for grid search
svr_param_grid = {
    'estimator__C': [0.1, 1, 10, 100],  # Regularization parameter
    'estimator__epsilon': [0.01, 0.1, 1],  # Epsilon in the epsilon-SVR model
    'estimator__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Specifies the kernel type to be used in the algorithm
}

# Perform grid search with cross-validation to find the best hyperparameters
svr_grid_search = GridSearchCV(svr_regressor, svr_param_grid, cv=5, n_jobs=-1, verbose=2)  # Initialize the GridSearchCV
start_time = time.time()  # Note the start time
svr_grid_search.fit(X_train, y_train)  # Fit the model to the training data
end_time = time.time()  # Note the end time

print("Time taken for grid search: {} seconds".format(end_time - start_time))  # Print the time difference

# Print the best hyperparameters found in the grid search
best_svr_params = svr_grid_search.best_params_  # Get the best parameters
print("Best SVR parameters: ", best_svr_params)  # Print the best parameters
print("Best score found: ", svr_grid_search.best_score_) # Print the best score

# Initialize an SVR model with the best hyperparameters
best_svr = SVR(kernel=best_svr_params['estimator__kernel'], C=best_svr_params['estimator__C'], epsilon=best_svr_params['estimator__epsilon'])  # Initialize the best SVR model

# Initialize the multioutput regressor with the best SVR model
svr_regressor = MultiOutputRegressor(best_svr)  # Wrap the best SVR model to handle multiple outputs

# Print cross-validation results
cv_results_svr = svr_grid_search.cv_results_  # Get the cross-validation results
cv_results_svr_df = pd.DataFrame(cv_results_svr)  # Convert the results to a DataFrame
cv_results_svr_df_sorted = cv_results_svr_df.sort_values(by='mean_test_score', ascending=False)  # Sort the results by mean test score in descending order

# Print complete SVR cross-validation results 
print("Complete SVR cross-validation results: ")
print(cv_results_svr_df_sorted)  # Print the sorted dataframe 

# Calculate and print the average mean test score for each kernel 
kernels = ['linear', 'rbf', 'poly', 'sigmoid']  # List of kernel types
average_scores = []  # Empty list to store average scores for each kernel type


# Loop through each kernel type
for kernel in kernels:
    kernel_results = cv_results_svr_df[cv_results_svr_df['param_estimator__kernel'] == kernel]  # Filter results for each kernel
    average_score = kernel_results['mean_test_score'].mean()  # Calculate mean test score for each kernel
    average_scores.append([kernel, average_score])  # Append the kernel type and average score to the list

average_scores_df = pd.DataFrame(average_scores, columns=['Kernel', 'Average Mean Test Score'])  # Convert list to a dataframe
print(average_scores_df)  # Print the dataframe

# Fit the SVR regressor to the training data
svr_regressor.fit(X_train, y_train)  # Fit the SVR regressor to the training data

# Predict the target variables for the test data
y_pred_svr = svr_regressor.predict(X_test)  # Predict the target variables for the test data

# Initialize a list to store accuracy results for the SVR model
svr_accuracies = []  # Initialize an empty list

# Number of runs for stability test
num_runs = 10

# Store scores of each run
run_scores = []

# Running the model 'num_runs' times
for run in range(num_runs):
    # Shuffle the data
    X, y = shuffle(X, y, random_state=run)

    # Split the data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=run)  # Split the data

    # Normalize the features to a range of [0, 1]
    scaler = MinMaxScaler()  # Initialize the MinMaxScaler
    X_train = scaler.fit_transform(X_train)  # Fit the scaler to the training data and transform it
    X_test = scaler.transform(X_test)  # Transform the testing data with the scaler

    # Fit the SVR regressor to the training data
    svr_regressor.fit(X_train, y_train)  # Fit the SVR regressor to the training data

    # Predict the target variables for the test data
    y_pred_svr = svr_regressor.predict(X_test)  # Predict the target variables for the test data

    # Calculate accuracy for the SVR model and append to the list
    svr_accuracy = mean_absolute_error(y_test, y_pred_svr)  
    run_scores.append(svr_accuracy)  # Append the accuracy to the list

# Calculate and print the variance and standard deviation of the performance scores across different runs
print("Variance of performance scores across runs:", np.var(run_scores))
print("Standard deviation of performance scores across runs:", np.std(run_scores))


# Loop over different training periods
for period in range(100, len(X_train), 100):  # Start with a training set of size 100 and increase by 100 in each iteration
    # Train the SVR model
    svr_regressor.fit(X_train[:period], y_train[:period])  # Train the SVR model on the first 'period' samples of the training data

    # Make predictions using the SVR model
    y_pred_svr = svr_regressor.predict(X_test)  # Make predictions on the test data

    # Calculate accuracy for the SVR model and append to the list
    svr_accuracy = mean_absolute_error(y_test, y_pred_svr)  # Calculate the mean absolute error between the actual and predicted values
    svr_accuracies.append(svr_accuracy)  # Append the accuracy to the list

# Plot accuracy over different training periods for the SVR model
plt.figure(figsize=(10, 5))  # Initialize a new figure
plt.plot(range(100, len(X_train), 100), svr_accuracies, label='SVR')  # Plot the accuracies
plt.xlabel('Training Period')  # Set the x-axis label
plt.ylabel('Mean Absolute Error')  # Set the y-axis label
plt.legend()  # Display the legend
plt.show()  # Show the plot

plt.figure(figsize=(10, 5))  # Set up a new plot of size 10x5
for kernel in kernels:  # Loop through each kernel in the list of kernels
    kernel_results = cv_results_svr_df[cv_results_svr_df['param_estimator__kernel'] == kernel]  # Get the results for the current kernel
    plt.hist(kernel_results['mean_test_score'], bins=20, alpha=0.5, label=f'{kernel}')  # Plot a histogram of the mean test scores for the current kernel

plt.xlabel('Mean Test Score')  # Set the x-axis label of the plot
plt.ylabel('Frequency')  # Set the y-axis label of the plot
plt.legend()  # Display the legend of the plot
plt.show()  # Show the plot

mse_svr = mean_squared_error(y_test, y_pred_svr)  # Calculate the mean squared error for the SVR model
mae_svr = mean_absolute_error(y_test, y_pred_svr)  # Calculate the mean absolute error for the SVR model
print('SVR Mean Squared Error:', mse_svr)  # Print the mean squared error for the SVR model
print('SVR Mean Absolute Error:', mae_svr)  # Print the mean absolute error for the SVR model

price_trends_pred = np.sign(y_pred_svr[:, 3] - y_pred_svr[:, 0])  # Predict the price trends (increase or decrease)
price_trends_actual = np.sign(y_test[:, 3] - y_test[:, 0])  # Get the actual price trends

accuracy = np.mean(price_trends_pred == price_trends_actual)  # Calculate the accuracy of the price trend prediction
precision = np.mean(price_trends_pred[price_trends_actual == 1] == 1)  # Calculate the precision of the price trend prediction
recall = np.mean(price_trends_pred[price_trends_pred == 1] == 1)  # Calculate the recall of the price trend prediction
f1_score = 2 * (precision * recall) / (precision + recall)  # Calculate the F1 score of the price trend prediction

print('Price Trend Prediction Accuracy:', accuracy)  # Print the accuracy of the price trend prediction
print('Price Trend Prediction Precision:', precision)  # Print the precision of the price trend prediction
print('Price Trend Prediction Recall:', recall)  # Print the recall of the price trend prediction
print('Price Trend Prediction F1-score:', f1_score)  # Print the F1 score of the price trend prediction

df_test = pd.DataFrame(y_test, columns=['Open', 'High', 'Low', 'Close'])  # Create a DataFrame with the test data
df_test['Predicted_Trend'] = price_trends_pred  # Add the predicted price trends to the DataFrame
df_test['Actual_Trend'] = price_trends_actual  # Add the actual price trends to the DataFrame

df_test['Signal'] = np.where(df_test['Predicted_Trend'] == 1, 1, 0)  # Buy when the predicted trend is up and hold when it is down
df_test['Returns'] = df_test['Signal'].shift() * df_test['Close'].pct_change()  # Calculate the returns based on the trading signals
df_test.dropna(inplace=True)  # Drop the NaN values

#T esting the ability of Cumulative Returns and Benchmark Returns
df_test['Cumulative_Returns'] = (1 + df_test['Returns']).cumprod() - 1  # Calculate the cumulative returns
df_test['Benchmark_Returns'] = df_test['Close'].pct_change().cumsum() + 1  # Calculate benchmark returns

print('Trading Strategy Returns:', df_test['Cumulative_Returns'].iloc[-1])  # Print the returns of the trading strategy
print('Benchmark Returns:', df_test['Benchmark_Returns'].iloc[-1])  # Print the benchmark returns

# Calculate and print the standard deviation of mean test score for each kernel 
kernel_stds = []  # Empty list to store standard deviations for each kernel type

# Loop through each kernel type
for kernel in kernels:
    kernel_results = cv_results_svr_df[cv_results_svr_df['param_estimator__kernel'] == kernel]  # Filter results for each kernel
    kernel_std = kernel_results['mean_test_score'].std()  # Calculate standard deviation of mean test score for each kernel
    kernel_stds.append([kernel, kernel_std])  # Append the kernel type and standard deviation to the list

kernel_stds_df = pd.DataFrame(kernel_stds, columns=['Kernel', 'Std of Mean Test Score'])  # Convert list to a dataframe
print(kernel_stds_df)  # Print the dataframe

# Plot standard deviation of mean test score for each kernel 
plt.figure(figsize=(10, 5))  # Initialize a new figure
plt.bar(kernel_stds_df['Kernel'], kernel_stds_df['Std of Mean Test Score'])  # Bar plot of standard deviation of mean test score for each kernel 
plt.xlabel('Kernel')  # Set the x-axis label
plt.ylabel('Standard Deviation of Mean Test Score')  # Set the y-axis label
plt.title('Kernel Stability')  # Set the title
plt.show()  # Show the plot

# Plot the actual vs predicted close values
plt.figure(figsize=(10, 5))
sns.lineplot(data=pd.DataFrame({'Actual': y_test[:, 3], 'Predicted': y_pred_svr[:, 3]}))
plt.title('Actual vs Predicted Close Prices')
plt.xlabel('Sample')
plt.ylabel('Price')
plt.legend()
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

# Create a DataFrame with future predicted prices and display it
future_prices_df = pd.DataFrame(y_future_pred_svr, columns=['Open', 'High', 'Low', 'Close'])  # Create DataFrame with predicted prices
future_prices_df.index = future_dates  # Set the index to future dates
print(future_prices_df)  # Print the DataFrame

# Define dictionaries to hold different metrics for each kernel
train_sizes, train_scores, test_scores = {}, {}, {}  # Initialize dictionaries to store sizes and scores

# Loop over different kernels to train and evaluate the model
for kernel in kernels:  
    # Initialize SVR with the optimal parameters
    best_svr = SVR(kernel=kernel, C=best_svr_params['estimator__C'], epsilon=best_svr_params['estimator__epsilon'])  # Initialize SVR model with best params
    svr_regressor = MultiOutputRegressor(best_svr)  # Use MultiOutputRegressor for handling multiple outputs

    # Calculate learning curve with 5-fold cross-validation
    sizes, train_score, test_score = learning_curve(svr_regressor, X_train, y_train, cv=5, scoring='neg_mean_squared_error')  # Calculate learning curve
    
    # Store the calculated metrics for each kernel
    train_sizes[kernel] = sizes  # Store sizes for each kernel
    train_scores[kernel] = train_score  # Store training scores for each kernel
    test_scores[kernel] = test_score  # Store test scores for each kernel

# Plot the learning curves
plt.figure(figsize=(10, 5))  # Set the size of the figure

# Loop over different kernels to plot the learning curves
for kernel in kernels:  
    # Plot training and validation errors for each kernel
    plt.plot(train_sizes[kernel], -np.mean(train_scores[kernel], axis=1), label=f"Training error - {kernel}")  # Plot training error
    plt.plot(train_sizes[kernel], -np.mean(test_scores[kernel], axis=1), label=f"Validation error - {kernel}")  # Plot validation error

# Add labels, title, legend, and grid to the plot
plt.xlabel("Training set size", fontsize=14)  # Add x-axis label
plt.ylabel("MSE", fontsize=14)  # Add y-axis label
plt.title("Learning curves", fontsize=16)  # Add title
plt.legend()  # Add legend
plt.grid()  # Add grid
plt.show()  # Display the plot
