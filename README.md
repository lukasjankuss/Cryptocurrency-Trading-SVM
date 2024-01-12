# Cryptocurrency Trading: Support Vector Machine

## Abstract
This research investigates the application of Support Vector Machine (SVM) kernels in predicting cryptocurrency market trends. The study identifies significant variations in prediction accuracy across different SVM kernels and cryptocurrencies, emphasizing the role of optimal parameter tuning. Despite the noted limitations, including a restricted sample of cryptocurrencies and a sole focus on SVM models, the research concludes that SVM kernels significantly improve prediction accuracy. This work provides a foundation for future advancements in machine learning applications within financial market analysis.

# Project Setup and Launch Instructions

This document contains the necessary instructions to set up and launch the provided project. The author affirms that they are the sole creator of the programs contained within this archive, except where explicitly indicated otherwise.

Date: 04/08/2023

## 1. Unzipping the Project Files
To begin, decompress the received zipped project files using your operating system's built-in decompression tool. Typically, this can be done by right-clicking on the zipped file and selecting "Extract All" or a similar option.

## 2. Installing Python and Required Libraries
The script is developed in Python. If Python is not installed on your system, it can be downloaded from the [official Python website](https://www.python.org/downloads/). Additionally, several Python libraries including `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `seaborn` are required. These can be installed using the pip package manager with the following command: pip install pandas numpy scikit-learn matplotlib seaborn

## 3. Modifying the Script for Alternative Data Usage
By default, the script utilizes Ethereum data (ETH-USD.csv). To process different datasets (e.g., Bitcoin, Cardano), the script must be modified. Replace the path in the line `df = pd.read_csv('data/Ethereum/ETH-USD.csv')` with the path to your desired dataset.

## 4. Executing the Script
Execute the script by following these steps:
- Open a terminal or command prompt.
- Navigate to the directory containing the `KernelPerformance.py` script using the `cd` command.
- Run the script with the command `python KernelPerformance.py`.
This will initiate the script's data processing and output generation.

## 5. Analyzing the Output
The script outputs extensive information to the console, including optimal hyperparameters, model performance, and various metrics and statistics. It also generates plots illustrating the model's performance and results.

## 6. Interpreting the Results
The script employs a Support Vector Regression (SVR) model with a grid search methodology for hyperparameter optimization. It calculates and reports metrics such as Mean Squared Error, Mean Absolute Error, Precision, Recall, and F1-score for model evaluation.

## 7. Conclusion
Upon completing the script execution and result analysis, the terminal window can be closed. Note that the script's grid search with cross-validation is computationally intensive, which may prolong execution time. Several pop-up windows will display results sequentially. For convenience, individual results for each cryptocurrency are included within their respective data folders.

# Project Structure

The project is organized as follows:

FinalProject21201706/
├── data/
│   ├── Bitcoin/
│   │   ├── Bitcoin.txt
│   │   └── BTC-USD.csv
│   ├── Cardano/
│   │   ├── Cardano.txt
│   │   └── ADA-USD.csv
│   ├── Dogecoin/
│   │   ├── Dogecoin.txt
│   │   └── DOGE-USD.csv
│   ├── Ethereum/
│   │   ├── Ethereum.txt
│   │   └── ETH-USD.csv
│   ├── Litecoin/
│   │   ├── Litecoin.txt
│   │   └── LTC-USD.csv
│   ├── Ripple/
│   │   ├── Ripple.txt
│   │   └── XRP-USD.csv
│   └── Tether/
│       ├── Tether.txt
│       └── USDT-USD.csv
├── KernelPerformance.py
└── README.md
