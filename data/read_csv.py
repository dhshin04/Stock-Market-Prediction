import os
import pandas as pd
import torch
from config import lag, ma_window


def get_stocks():
    # Returns list of csv file names in stocks folder except __init__.py
    stocks_list = os.listdir(os.path.join(os.path.dirname(__file__), 'stocks'))
    for i, stock in enumerate(stocks_list):
        if stock == '__init__.py':
            stocks_list.pop(i)
            break
    return stocks_list


def retrieve_sample():
    '''
    Retrieve stock samples chosen by Gemini

    Returns:
        stock_sample (list): List of stocks selected by Gemini with ".csv" appended
    '''
    stocks_file_path = os.path.join(os.path.dirname(__file__), 'stocks.txt')
    with open(stocks_file_path, 'r') as stocks_file:
        stock_sample = stocks_file.read().split('\n')
        
        for i in range(len(stock_sample)):
            stock_sample[i] += '.csv'
        
        # Handle improper stock suggestion(s) given by Gemini
        stocks_list = get_stocks()
        for i, stock in enumerate(stock_sample):
            if stock not in stocks_list:
                stock_sample.pop(i)

    return stock_sample


def remove_noise(dataframe, lag=5, ma_window=10):
    '''
    Use differencing and moving average to remove noise while preserving trend

    Arguments:
        dataframe (pd.DataFrame): Original dataframe without noise removal
        lag (int): Size of lag for differencing
        ma_window (int): Window size to compute moving average
    Returns:
        new_df (pd.DataFrame): New dataframe with noise removed
    '''
    new_df = pd.DataFrame(index=dataframe.index)
    for column in dataframe:
        # Get series
        series = dataframe[column]

        # Find series(t - lag)
        series_lag = series.shift(lag)

        # Calculate differenced series
        series_diff = series - series_lag

        # Calculate moving averages of lagged series and differenced series
        ma_lag = series_lag.rolling(window=ma_window).mean()
        ma_diff = series_diff.rolling(window=ma_window).mean()

        # Add the two moving averages and add to dataframe
        new_df[column] = ma_lag + ma_diff
    
    return new_df.dropna()      # Remove NaN that could arise from differencing or moving average


def train_test_split(csv_files, train_split, test_split):
    '''
    Converts csv files into lists of series tensors. 
    Then, split each tensor into train, validation, and test series.

    Arguments:
        csv_file (list): List of csv file names
        train_split (str): Date to split training set from val/test sets
        test_split (str): Date to split test set from validation set
    Returns:
        (tuple): Train, validation, and test series each as lists of tensors
    '''
    stocks_path = os.path.join(os.path.dirname(__file__), 'stocks')

    train_list = []
    val_list = []
    test_list = []
    for csv_file in csv_files:
        stock_path = os.path.join(stocks_path, csv_file)

        stock_data = pd.read_csv(stock_path, index_col='Date', parse_dates=['Date'])
        stock_data = remove_noise(stock_data, lag, ma_window)
        series = torch.tensor(stock_data.values, dtype=torch.float32)

        train_split_idx = stock_data.index.get_loc(train_split)
        test_split_idx = stock_data.index.get_loc(test_split)

        train_series = series[:train_split_idx]
        val_series = series[train_split_idx:test_split_idx]
        test_series = series[test_split_idx:]

        # Remove rows that does not have a number (no data for the day)
        train_series = train_series[~torch.any(train_series.isnan(), dim=1)]
        val_series = val_series[~torch.any(val_series.isnan(), dim=1)]
        test_series = test_series[~torch.any(test_series.isnan(), dim=1)]
        
        train_list.append(train_series)
        val_list.append(val_series)
        test_list.append(test_series)

    return train_list, val_list, test_list
