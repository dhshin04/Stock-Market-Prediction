import os
import pandas as pd
import torch


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


def csv_to_tensor(csv_files):
    '''
    Converts csv files into 2D tensors

    Arguments:
        csv_file (list): List of csv file names
    Returns:
        stock_data (list): List of tensors that csv files converted to
    '''
    stocks_path = os.path.join(os.path.dirname(__file__), 'stocks')

    stock_list = []
    for csv_file in csv_files:
        stock_path = os.path.join(stocks_path, csv_file)

        stock_data = pd.read_csv(stock_path, index_col='Date', parse_dates=['Date'])
        stock_data = torch.tensor(stock_data.values, dtype=torch.float32)

        stock_list.append(stock_data)

    return stock_list


if __name__ == '__main__':
    stock_sample = retrieve_sample()
    csv_files = [stock_sample[0]]
    stock_list = csv_to_tensor(csv_files)
    print(stock_list[0].shape)
