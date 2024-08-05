''' Choose n tech stocks from available stocks list using Gemini '''

from dotenv import load_dotenv
import google.generativeai as genai
import sys, os
import pandas as pd

sys.path.append(os.path.abspath('..'))
from config import start_date

NUM_STOCKS = 20
START_DATE = 'before January 2000'
END_DATE = 'April 2020'

# Load API KEY
load_dotenv()
genai.configure(api_key=os.getenv('GEM_API_KEY'))

# Define Model
model = genai.GenerativeModel('gemini-1.5-flash')

# Generate Response
response = model.generate_content(
    f'''
    Provide a list of {NUM_STOCKS} major stocks that have been trading on NASDAQ 
    {START_DATE} and continued trading until {END_DATE}. The stocks must 
    represent the entire tech industry across multiple fields. Please also double check
    to make sure the stocks were traded in this range. Print the stocks 
    in this format without any other words in the response, sorted in alphabetical order: 
    
    XXXX
    XXXX
    ...
    '''
).text

# Trim whitespace and newline added at end of every Gemini response
response = response[:-2]
stocks = response.split('\n')

# Data Cleaning - filter out stocks that are out of range or are not in dataset
stocks_path = os.path.join(os.path.dirname(__file__), 'stocks')
stocks_list = os.listdir(stocks_path)     # List of stock names
valid_stocks = []

for stock in stocks:
    stock_file = stock + '.csv'
    if stock_file not in stocks_list:
        continue

    stock_path = os.path.join(stocks_path, stock_file)
    stock_data = pd.read_csv(stock_path, index_col='Date', parse_dates=['Date'])

    if start_date in stock_data.index:
        valid_stocks.append(stock)

updated_response = ''
for stock in valid_stocks:
    updated_response += stock + '\n'
updated_response = updated_response[:-1]    # Remove last empty line

# Write response to stocks.txt
stocks_file_path = os.path.join(os.path.dirname(__file__), 'stocks.txt')
with open(stocks_file_path, 'w') as stocks_file:
    stocks_file.write(updated_response)
