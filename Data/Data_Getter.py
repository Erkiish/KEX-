import pandas as pd 
import pickle
import time
import os

if __name__ == '__main__':
    from Nasdaq_API import nasdaq_get_history
else:
    from Data.Nasdaq_API import nasdaq_get_history
""" 
Missing data (oliver):
BHG
SAND
SBB_D
ERIC_B

"""

DATA_STORAGE_LOCATION = {
                        "oliver": "/Users/oliver/Desktop/KEX-ARBETE/Data_Storage",
                        "erik": ""
}


def id_getter() -> pd.DataFrame:

    with open('Mapping_Data/Ticker_Lista_Sektor_Bransch_ticker_intersection_2022-01-20.pkl', 'rb') as file:
        mapping_file = pickle.load(file)
    
    return mapping_file.loc[mapping_file['lista'] == 'Large Cap']


def data_downloader(user: str):

    start_date = '1900-01-01'
    end_date = '2023-01-01'

    id_map = id_getter().reset_index()

    storage_location = DATA_STORAGE_LOCATION[user]
    file_name = storage_location + '/STOCK_DATA.xlsx'

    with pd.ExcelWriter(file_name) as writer:

        for id_tuple in id_map.itertuples():

            ticker = id_tuple.ticker.replace(' ', '_')
            nasdaq_id = id_tuple.nasdaq_id
            try:
                ticker_data = nasdaq_get_history(nasdaq_id, start_date, end_date)
            except:
                print(ticker)
            
            ticker_data.to_excel(writer, sheet_name=ticker)

            time.sleep(3)


def create_optimized_data_set(user: str):
    """
    Function for generating an optimized, easier to handle data-set for performing analysis.
    The generated data from this function is used for the class DataClass in the Data_Class module.

    Args:
        user (str): Name of the user accessing this function. For locating the saved excel file generated from the function
                    data_downloader.
        
    """

    data = load_data(user)

    time_interval_dict = {}
    df_list = []

    for ticker, df in data.items():
        df = df.rename({'close':ticker}, axis=1).set_index('date')[ticker]
        min_date = df.index.min()
        max_date = df.index.max()

        time_interval_dict[ticker] = {'min_date':min_date, 'max_date':max_date}
        df_list.append(df)
    
    mega_df = pd.concat(df_list, axis=1)
    mega_df.to_excel(DATA_STORAGE_LOCATION[user] + '/OPTIMIZED_DATA_SET.xlsx')
    
    time_interval_df = pd.DataFrame.from_dict(time_interval_dict, orient='index')
    time_interval_df.index.name = 'ticker'
    time_interval_df.to_excel(DATA_STORAGE_LOCATION[user] + '/TIME_INTERVALS.xlsx')
        

def load_data(user: str) -> dict[str, pd.DataFrame]:
    """
    Loads data from excel-sheet into specified time_frame and returns 


    Args:
        user (str): [description]
        time_frame (str): [description]

    Returns:
        pd.DataFrame: [description]
    """
    file_loc = DATA_STORAGE_LOCATION[user] + '/STOCK_DATA.xlsx'
    return pd.read_excel(file_loc, sheet_name=None, parse_dates=['date'])


if __name__ == '__main__':
    create_optimized_data_set('oliver')


