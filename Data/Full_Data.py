try:
    from Data_Getter import load_data, DATA_STORAGE_LOCATION
    from Data_Cleaning import nan_handler, zero_handler
except:
    from Data.Data_Getter import load_data, DATA_STORAGE_LOCATION
    from Data.Data_Cleaning import nan_handler, zero_handler
import pandas as pd
from typing import Union, Dict

class FullData:

    def __init__(self, user: str) -> None:

        self.all_data_uncleaned = load_data(user=user)

        self.time_interval = pd.read_excel(DATA_STORAGE_LOCATION[user] + '/TIME_INTERVALS.xlsx', parse_dates=['min_date', 'max_date']).set_index('ticker')
        
        self.all_data_cleaned = self._clean_data()
    
    def _clean_data(self) -> Dict[str, pd.DataFrame]:

        all_df_dict = {}
        df_len_dict = {}

        for ticker, df in self.all_data_uncleaned.items():

            df = zero_handler(nan_handler(df))
            df['ticker'] = ticker

            df_len_dict[ticker] = len(df)

            all_df_dict[ticker] = df

        self.df_len = pd.DataFrame.from_dict(df_len_dict, orient='index')

        return all_df_dict

    def get_data(self, start_date: str='1900-01-01', end_date: str='2050-01-01', min_len: int=0, tickers: Union[list, bool]=False) -> Dict[str, pd.DataFrame]:
        """Method for getting data based on specifications of dates, minimum length and tickers.

        Args:
            start_date (str, optional): [description]. Defaults to '1900-01-01'.
            end_date (str, optional): [description]. Defaults to '2050-01-01'.
            min_len (int, optional): [description]. Defaults to 0.
            tickers (Union[list, bool], optional): [description]. Defaults to False.

        Returns:
            dict[str, pd.DataFrame]: [description]
        """
        self.tickers = tickers

        if not tickers:
            self.tickers = list(self.all_data_cleaned.keys())
            data = self.all_data_cleaned.copy()
        else:
            data = {ticker:self.all_data_cleaned[ticker] for ticker in self.tickers}
        
        if min_len != 0:
            tickers_x = list((self.df_len > min_len).index)
            self.tickers = list(set(tickers_x) & set(self.tickers))
            data = {ticker:data[ticker] for ticker in self.tickers}
        
        if start_date != '1900-01-01' or end_date != '2050-01-01':
            tickers_x = list(self.time_interval.loc[(self.time_interval['min_date'] <= start_date) & (self.time_interval['max_date'] >= end_date)].index)
            tickers = list(set(tickers_x) & set(self.tickers))
            data = {ticker:data[ticker].loc[start_date:end_date, :] for ticker in self.tickers}

        return data