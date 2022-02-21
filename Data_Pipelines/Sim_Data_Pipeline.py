
from Strategies.Strategies import *
from Data.Full_Data import FullData
from Data.GBM import GBM
import pandas as pd 

def indicator_adder_x(data: pd.DataFrame) -> pd.DataFrame:

    data = add_rsi(data, len=14)
    data = add_sma(data, len=200)
    data = add_sma(data, 30)
    return add_ema(data, len=20)


def sim_data_getter_x(n_stocks: int, len: int, init_value: int=100) -> dict[str, pd.DataFrame]:

    gbm_class = GBM(init_value=init_value)
    return {
        str(stock): indicator_adder_x(gbm_class.generate_ohlcv_data(length=len))
        for stock in range(n_stocks)
    }


def test_pipeline_x(n_stocks: int, len: int, strategy: object) -> dict[str, pd.DataFrame]:

    data = sim_data_getter_x(n_stocks, len)

    return {ticker:strategy.get_sell_signals(strategy.get_buy_signals(data[ticker])) for ticker in data.keys()}



