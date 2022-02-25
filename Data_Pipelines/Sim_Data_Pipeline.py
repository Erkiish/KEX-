
from Strategies.Strategies import *
from Data.Full_Data import FullData
from Data.GBM import GBM
import pandas as pd 
from dataclasses import dataclass
from typing import Union, Tuple
from scipy.stats import uniform
from random import shuffle



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



def test_pipeline_x(batch_size: int, time_steps: int, strategy: object, scale_handler: ScalerHandler=ScalerHandler([], [], []), test_data: bool=False) -> Union[Tuple[np.ndarray, dict], np.ndarray]:
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
    

def time_series_resampler(data: np.ndarray, target_col: int, series_length: int, class_pct: dict[int, float]) -> np.ndarray:
    """ 'Resamples' and converts numpy dataset into specified length and class weights.

    Args:
        data (np.ndarray): The dataset to convert and resample from.
        target_col (int): The index for the column containing the targets/classes of each timestep.
        length (int): Wanted length of each sample.
        class_weight (dict[int, float]): A dict containing the weights of each class, used for resampling.

    Returns:
        np.ndarray: An ndarray with n ammount of samples with length series_length, same number of features as data and a composition of the classes
                    as specified in the class_pct.
    """
    all_samples = []
    for sample in range(data.shape[0]):

        sample_data = data[sample]
        nbr_of_ones = (sample_data[:, target_col] == 1)
        if nbr_of_ones.sum() == 0: 
            continue

        # Generates sequences with 1 as target
        class_one_index = np.asarray(nbr_of_ones).nonzero()

        nbr_one_targets_sample = 0
        for index in class_one_index:
            index = index[0]
            new_data = sample_data[:index+1]
            new_data_len = new_data.shape[0]
            if new_data_len < series_length:
                continue
            new_one_data = new_data[-series_length:]
            nbr_one_targets_sample += 1
            all_samples.append(new_one_data)
        
        tot_samples = nbr_one_targets_sample/class_pct[1]

        # Generates sequences with 0 as target, based on the class_pct input.
        data_len = sample_data.shape[0]
        index_available = data_len - series_length
        req = round(tot_samples*class_pct[0])
        nbr_zero_samples = 0
        while nbr_zero_samples < req:

            index_start = round(uniform.rvs(loc=0, scale=index_available))
            index_end = index_start + series_length
            new_zero_data = sample_data[index_start:index_end]

            if new_zero_data[-1, target_col] == 1:
                continue

            all_samples.append(new_zero_data)
            nbr_zero_samples += 1
    
    return np.stack(all_samples).astype(np.float32)












