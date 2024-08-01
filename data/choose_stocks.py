''' Choose 25 unique tech stocks from available stocks list '''

from dotenv import load_dotenv
import google.generativeai as genai
import os

NUM_STOCKS = 10
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
    represent the entire tech industry across multiple fields. Print the stocks 
    in this format without any other words in the response, sorted in alphabetical order: 
    
    XXXX
    XXXX
    ...
    '''
).text

# Trim whitespace and newline added at end of every Gemini response
response = response[:-2]

# Write response to stocks.txt
stocks_file_path = os.path.join(os.path.dirname(__file__), 'stocks.txt')
with open(stocks_file_path, 'w') as stocks_file:
    stocks_file.write(response)
