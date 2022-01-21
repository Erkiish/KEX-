import pandas as pd

if __name__ == '__main__':
    from Data_Getter import load_data, DATA_STORAGE_LOCATION
else:
    from Data.Data_Getter import load_data, DATA_STORAGE_LOCATION
class DataSetGenerator:
    """
    DataSetGenerator is a dataset generator that smoothly returns conformed data for a choosen period and time-frame.
    It uses the file OPTIMIZED_DATA_SET that has to be generated before, in order to quickly create new datasets.
    """

    def __init__(self, user: str):
        """
        Initializes DataSetGenerator.

        Args:
            user (str): Name of user, needed to access the OPTIMIZED_DATA_SET excel file. 
        """
        
        self.data = pd.read_excel(DATA_STORAGE_LOCATION[user] + '/OPTIMIZED_DATA_SET.xlsx', parse_dates=['date']).set_index('date')
        self.time_interval = pd.read_excel(DATA_STORAGE_LOCATION[user] + '/TIME_INTERVALS.xlsx', parse_dates=['min_date', 'max_date']).set_index('ticker')
    
    def get_weekly_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        get_weekly_data returns a pandas DataFrame with a weekly frequency between the choosen start- and end-date.

        Args:
            start_date (str): earliest date of the dataset
            end_date (str): latest date of the dataset

        Returns:
            pd.DataFrame: pandas DataFrame with the choosen frequency and timeframe.
        """
        
        weekly_data = self._conform(start_date, end_date)

        return weekly_data.resample('W-FRI').ffill()

    def get_monthly_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        get_monthly_data returns a pandas DataFrame with a monthly frequency between the choosen start- and end-date.

        Args:
            start_date (str): earliest date of the dataset
            end_date (str): latest date of the dataset

        Returns:
            pd.DataFrame: pandas DataFrame with the choosen frequency and timeframe.
        """
        
        monthly_data = self._conform(start_date, end_date)

        return monthly_data.resample('M').ffill()
    
    def get_daily_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        get_daily_data returns a pandas DataFrame with a daily frequency between the choosen start- and end-date.

        Args:
            start_date (str): earliest date of the dataset
            end_date (str): latest date of the dataset

        Returns:
            pd.DataFrame: pandas DataFrame with the choosen frequency and timeframe.
        """

        return self._conform(start_date, end_date)

    def _conform(self, start_date: str, end_date: str):

        tickers = list(self.time_interval.loc[(self.time_interval['min_date'] <= start_date) & (self.time_interval['max_date'] >= end_date)].index)

        return self.data.loc[start_date:end_date, tickers]


if __name__ == '__main__':

    dataclass = DataSetGenerator('oliver')

    print(dataclass.get_monthly_data('2012-01-01', '2020-01-01'))