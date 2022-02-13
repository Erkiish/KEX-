
from Strategies.Strategies import *
from Data.Full_Data import FullData
import pandas as pd 

def TestPipeline(user: str, data_settings: dict, strategy: object) -> dict[str, pd.DataFrame]:

    raw_data_class = FullData(user)
    raw_data = raw_data_class.get_data(**data_settings)

    return {ticker:strategy.get_sell_signals(strategy.get_buy_signals(raw_data[ticker])) for ticker in raw_data_class.tickers}






