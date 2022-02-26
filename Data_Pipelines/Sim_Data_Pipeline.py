
from Strategies.Strategies import *
from Data.GBM import GBM
import pandas as pd 
from dataclasses import dataclass
from typing import Union, Tuple, Callable
from scipy.stats import uniform
import pandas_ta as ta



@dataclass
class ScaleHandler:

    min_max_scaler_cols: list[Tuple]
    standardize_scaler_cols: list[Tuple]
    divide_scaler_cols: dict[Union[float, str], Tuple]

    def scale(self, data: np.ndarray) -> np.ndarray:

        return self.divide_scaler(self.standardize_scaler(self.min_max_scaler(data)))
    
    def min_max_scaler(self, data: np.ndarray) -> np.ndarray:

        for set_ in self.min_max_scaler_cols:

            min_ = np.amin(data[:, set_])
            max_ = np.amax(data[:, set_])
            data[:, set_] = (data[:, set_] - min_)/(max_ - min_ + 0.00001)

        return data
    
    def min_max_reset(self, data: np.ndarray, reset_info: list) -> np.ndarray:

        for params in reset_info:

            data[:, params['cols']] = data[:, params['cols']]*(params['max'] - params['min'] + 0.00001) + params['min']
        
        return data

    def standardize_scaler(self, data: np.ndarray) -> np.ndarray:
        
        for set_ in self.standardize_scaler_cols:

            stacked_data = data[:, set_]
            mean = np.mean(stacked_data)
            std = np.std(stacked_data)

            data[:, set_] = (data[:, set_] - mean)/(std)

        return data
    
    def standardize_reset(self, data: np.ndarray, reset_info: list) -> np.ndarray:

        for params in reset_info:

            data[:, params['cols']] = data[:, params['cols']]*params['std'] + params['mean']
        
        return data
    
    def divide_scaler(self, data: np.ndarray) -> np.ndarray:

        for div, set_ in self.divide_scaler_cols.items():

            if div == 'MAX':
                div = np.amax(data[:, set_])

            data[:, set_] = data[:, set_]/div
        
        return data
    
    def divide_reset(self, data: np.ndarray, reset_info: list) -> np.ndarray:

        for params in reset_info:

            data[:, params['cols']] = data[:, params['cols']]*params['div']
        
        return data


def indicator_adder_x(data: np.ndarray) -> np.ndarray:
    """Adds indicator data columns to the end of the passed in array.
    For now the transformation looks like this:
    data has n-columns, column n+1 is rsi_14, column n+2 is sma_200, column n + 3 is sma_30,
    column n + 4 is pct_change of close (column 3 in data which is an array of ohlcv),
    column n + 5 is ema_20

    Args:
        data (np.ndarray): numpy array with ohlcv structure

    Returns:
        np.ndarray: numpy array with ohlcv rsi_14 sma_200 sma_30 pct_change ema_20
    """

    indicators = np.stack([ 
    ta.rsi(pd.Series(data[:, 3]), length=14), 
    ta.sma(pd.Series(data[:, 3]), length=200),
    ta.sma(pd.Series(data[:, 3]), length=30),
    pd.Series(data[:, 3]).pct_change(),
    ta.ema(pd.Series(data[:, 3]), length=20)
    ], axis=-1)
    return np.concatenate([data, indicators], axis=1)

@dataclass
class DataPipeline:

    batch_size: int
    time_steps: int
    indicator_adder: Callable
    strategy: Strategy
    scale_handler: ScaleHandler

    def _GBM_data_getter(self, init_value: int=100) -> dict[str, np.ndarray]:

        gbm_class = GBM(init_value=init_value)
        return {
            str(stock): self.indicator_adder(gbm_class.generate_ohlcv_data(length=self.time_steps))
            for stock in range(self.batch_size)
        }


    def _buy_sell_scale_fixer(self, raw_data: dict) -> Tuple[list, list]:

        scaled_list = []
        unscaled_list = []
        for ticker in raw_data:
            # Gets the buy and sell data
            buy_sell_data = self.strategy.get_sell_signals(self.strategy.get_buy_signals(raw_data[ticker]))
            
            # Removes rows with nan-values
            buy_sell_data = buy_sell_data[~np.isnan(buy_sell_data).any(axis=1)]

            unscaled_list.append(buy_sell_data.copy())

             # Applies scale method and returnd ndarray
            scaled_list.append(self.scale_handler.scale(buy_sell_data))

        return scaled_list, unscaled_list


    def get_data(self, resample: Union[bool, dict]=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates 3D numpy array of simulated stockdata and strategy result features. Shape is (batch_size, time_steps - 2, n_features=depends on strategy)
        Where n_features represents ohlc data, volume data, indicator data and sell- and buy-data in binary (0 if nothing 1 if sell/buy).

        Args:
            resample Union[bool, dict]: False if no time-series resampling if wanted. Provide dict if wanted, with following structure:
                                        {target_col: int, series_length: int, class_pct: dict[int, float]}
        Returns:
            np.array: Returns a numpy array of roughly the size (batch_size, time_steps, n_features) which all depends on indicator, strategy and resampling.
                    Example: Rows with NaN-values are removed so depending on indicators x-ammount of time-steps will dissapear.
        """
        raw_data = self._GBM_data_getter()
        
        ndarray_list_scaled, ndarray_list_unscaled = self._buy_sell_scale_fixer(raw_data)

        data_scaled, data_unscaled = np.stack(ndarray_list_scaled).astype(np.float32), np.stack(ndarray_list_unscaled).astype(np.float32)

        if isinstance(resample, bool):
            return data_scaled, data_unscaled
        
        return self.time_series_resampler(data_scaled, **resample), data_unscaled
        


    def time_series_resampler(self, data: np.ndarray, target_col: int, series_length: int, class_pct: dict[int, float]) -> np.ndarray:
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












