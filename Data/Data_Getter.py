import pandas as pd 
import pickle
import time
import os
from Nasdaq_API import nasdaq_get_history
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


def load_data(user: str, time_frame: str) -> pd.DataFrame:
    """
    Loads data from excel-sheet into specified time_frame and returns 


    Args:
        user (str): [description]
        time_frame (str): [description]

    Returns:
        pd.DataFrame: [description]
    """

if __name__ == '__main__':
    #data_downloader(user='oliver')
    print('hej')

