# Remove __init__.py from list of stock names
import os


def remove_init(stocks_list):
    '''
    Removes __init__.py from given list of stock names

    Arguments:
        stocks_list (list): List of stock names
    '''
    
    for i, stock in enumerate(stocks_list):
        if stock == '__init__.py':
            stocks_list.pop(i)


stocks_list = os.listdir(os.path.join(os.path.dirname(__file__), 'stocks'))
remove_init(stocks_list)
