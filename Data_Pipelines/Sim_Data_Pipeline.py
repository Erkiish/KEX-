
from Strategies.Strategies import *
from Data.Full_Data import FullData
from Data.GBM import GBM
import pandas as pd 
from dataclasses import dataclass
from typing import Union



@dataclass
class ScalerHandler:

    min_max_scaler_cols: list[tuple]
    standardizer_scaler_cols: list[tuple]
    divider_scaler_cols: dict[Union[float, str], tuple]

    def scale(self, data: pd.DataFrame) -> pd.DataFrame:

        return self.divider_scaler(self.standardizer_scaler(self.min_max_scaler(data)))
    
    def min_max_scaler(self, data: pd.DataFrame) -> pd.DataFrame:

        for set_ in self.min_max_scaler_cols:

            min_ = data.loc[:, set_].min().min()
            max_ = data.loc[:, set_].max().max()
            data.loc[:, set_] = (data.loc[:, set_] - min_)/(max_ - min_ + 0.00001)
        
        return data

    def standardizer_scaler(self, data: pd.DataFrame) -> pd.DataFrame:
        
        for set_ in self.standardizer_scaler_cols:

            stacked_data = data.loc[:, set_]
            if len(set_) > 1:
                stacked_data = stacked_data.stack()
            mean = stacked_data.mean()
            std = stacked_data.std()

            data.loc[:, set_] = (data.loc[:, set_] - mean)/(std)

        return data
    
    def divider_scaler(self, data: pd.DataFrame) -> pd.DataFrame:

        for div, set_ in self.divider_scaler_cols.items():

            if div == 'MAX':
                div = data.loc[:, set_].max().max()

            data.loc[:, set_] = data.loc[:, set_]/div
        
        return data


def indicator_adder_x(data: pd.DataFrame) -> pd.DataFrame:

    data = add_rsi(data, len=14)
    data = add_sma(data, len=200)
    data = add_sma(data, 30)
    data = data.assign(pct_change=data['close'].pct_change())
    return add_ema(data, len=20)


def sim_data_getter_x(n_stocks: int, len: int, init_value: int=100) -> dict[str, pd.DataFrame]:

    gbm_class = GBM(init_value=init_value)
    return {
        str(stock): indicator_adder_x(gbm_class.generate_ohlcv_data(length=len))
        for stock in range(n_stocks)
    }



def test_pipeline_x(batch_size: int, time_steps: int, strategy: object, scale_handler: ScalerHandler=ScalerHandler([], [], []), test_data: bool=False) -> np.ndarray:
    """
    Generates 3D numpy array of simulated stockdata and strategy result features. Shape is (batch_size, time_steps - 2, n_features=depends on strategy)
    Where n_features represents ohlc data, volume data, indicator data and sell- and buy-data in binary (0 if nothing 1 if sell/buy).

    Args:
        batch_size (int): Can be seen as the number of stocks, is essentially the number of different simulations to be created.
        time_steps (int): The length of each simulation, can be seen as the number of trading days for each stock. This input is reduced 
                        by 2 to accomodate for the reduced size of buy/sell-data
        strategy (object): The strategy that is to be used to generate buy and sell-features/columns.

    Returns:
        np.array: Returns a numpy array of size (batch_size, time_steps - 2, n_features=depends on strategy). Basically batch_size number 
                        of tables stacked on top of each other with length time_steps.
    """
    data = sim_data_getter_x(batch_size, time_steps)
    
   

    if not test_data:

        df_list = [scale_handler.scale(strategy.get_sell_signals(strategy.get_buy_signals(data[ticker])).dropna().reset_index(drop=True)) for ticker in data.keys()]

        df_col_map = {name:index for index, name in zip(list(df_list[0].columns.get_indexer(list(df_list[0].columns))), list(df_list[0].columns))}

        return np.stack(df_list).astype(np.float32), df_col_map
    return [scale_handler.scale(strategy.get_sell_signals(strategy.get_buy_signals(data[ticker])).dropna().reset_index(drop=True)) for ticker in data.keys()], [strategy.get_sell_signals(strategy.get_buy_signals(data[ticker])).dropna().reset_index(drop=True) for ticker in data.keys()]
    

def data_conformer(data: np.ndarray, target_col: int, series_length: int, class_weight: dict[int, float]) -> np.ndarray:
    """Converts and conforms sequence-to-sequence data into sequence-to-vector output, that is one output per time-series.

    Args:
        data (np.ndarray): _description_
        target_col (int): _description_
        length (int): _description_
        class_weight (dict[int, float]): _description_

    Returns:
        np.ndarray: _description_
    """

    for sample in range(data.shape[0]):

        sample_data = data[sample]
        nbr_of_ones = (sample_data[:, target_col] == 1)
        if nbr_of_ones.sum() == 0: 
            continue
        class_one_index = np.asarray(nbr_of_ones).nonzero()

        for index in class_one_index:
            new_data = sample_data[:index+1]
            new_data_len = new_data.shape[0]
            if new_data_len < series_length:
                continue
            new_data = 






