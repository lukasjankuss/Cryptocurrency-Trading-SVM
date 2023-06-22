from tkinter import *
import pandas as pd
from tkinter import messagebox

# SVR results data
best_parameters = {'estimator__C': 100, 'estimator__epsilon': 0.01, 'estimator__kernel': 'linear'}
mean_squared_error = 4047.85949860518
mean_absolute_error = 36.0969745677168
price_trend_prediction_accuracy = 0.9153846153846154
price_trend_prediction_precision = 0.9381443298969072
price_trend_prediction_recall = 1.0
price_trend_prediction_f1_score = 0.9680851063829787
trading_strategy_returns = -1.055539428539231e+21
benchmark_returns = 705.761023582324

# Complete SVR cross-validation results:
# Given the size of the DataFrame, it's omitted here but you can include it in the same way.

# Creating main tkinter window/toplevel
master = Tk()

text = f"""
Best SVR parameters:  {best_parameters}
SVR Mean Squared Error: {mean_squared_error}
SVR Mean Absolute Error: {mean_absolute_error}
Price Trend Prediction Accuracy: {price_trend_prediction_accuracy}
Price Trend Prediction Precision: {price_trend_prediction_precision}
Price Trend Prediction Recall: {price_trend_prediction_recall}
Price Trend Prediction F1-score: {price_trend_prediction_f1_score}
Trading Strategy Returns: {trading_strategy_returns}
Benchmark Returns: {benchmark_returns}
"""

messagebox.showinfo("SVR Results", text)

# Infinite loop can be terminated by 
# keyboard or mouse interrupt
# or by any other method line mainframe.destroy()
mainloop()
