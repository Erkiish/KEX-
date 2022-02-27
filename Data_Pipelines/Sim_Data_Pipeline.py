
from Strategies.Strategies import *
from Data.GBM import GBM
import pandas as pd 
from dataclasses import dataclass
from typing import Union, Tuple, Callable, Dict, List
from scipy.stats import uniform
import pandas_ta as ta



@dataclass
class ScaleHandler:

    min_max_scaler_cols: List[Tuple]
    standardize_scaler_cols: List[Tuple]
    divide_scaler_cols: Dict[Union[float, str], Tuple]

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

    def _GBM_data_getter(self, init_value: int=100) -> Dict[str, np.ndarray]:

        gbm_class = GBM(init_value=init_value)
        return {
            str(stock): self.indicator_adder(gbm_class.generate_ohlcv_data(length=self.time_steps))
            for stock in range(self.batch_size)
        }


    def _buy_sell_scale_fixer(self, raw_data: Dict) -> Tuple[list, list]:

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


    def get_data_lite(self, resample: Union[bool, dict]=False) -> Tuple[np.ndarray, np.ndarray]:
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
    
    def get_data_full(self, un_resampled_data_fraction: float=0.333, target_col: int=10,
                         series_length: int=30, class_pct: Union[float, Dict]=None,
                         train_samples_fraction: float=0.6, feature_start_col: int=0,
                         feature_end_col: int=10
                         ) -> Dict[str, np.ndarray]:
        """ Method for generating the full scope of an ML data pipeline. Lots of presupposed paramaters for ease of use when lots of parameters make code unreadable.
        Can be seen as the final abstraction layer of an data-pipeline of an ML-project.
        Returns:


        Args:
            un_resampled_data_fraction (float, optional): Fraction of data to be remained unresampled, for use when validating trained model, see X/Y_unresample_pred. Defaults to 0.333.
            target_col (int, optional): The column of the dataset that has the correct classes for binary predictions, could be multiclass with input as a tuple. Defaults to 10.
            series_length (int, optional): Wanted length of each sample. Defaults to 30.
            class_pct (Union[float, Dict], optional): Dict containing class fractions for generating resampled data. Defaults to {0:0.9, 1:0.1}.
            train_samples_fraction (float, optional): The fraction of the data to be used for training of the model. Defaults to 0.6.
            feature_start_col (int, optional): The column where features for the X_data starts. Defaults to 0.
            feature_end_col (int, optional): The column where features for the X_data ends. Defaults to 10.

        Returns:
            Dict[str, np.ndarray]: {
                                    'X_train': np.ndarray,
                                    'Y_train': np.ndarray,
                                    'X_val': np.ndarray,
                                    'Y_val': np.ndarray,
                                    'X_unresampled_unscaled': np.ndarray,
                                    'X_resample_pred': np.ndarray,
                                    'Y_resample_pred': np.ndarray,
                                    'X_unresample_pred': np.ndarray,
                                    'Y_unresample_pred': np.ndarray
        }
        """
        # Parameter setup for time series resampling.
        if class_pct is None:
            class_pct = {0:0.9, 1:0.1}
        resample_params = {
                    'target_col': target_col,
                    'series_length': series_length, 
                    'class_pct': class_pct
        }
        # Gets scaled and unscaled data.
        data_scaled, data_unscaled = self.get_data_lite()
        # Saves unscaled and un_resampled_data for matching later on.
        un_resampled_data_samples = round(self.batch_size*un_resampled_data_fraction)
        un_resampled_data = data_scaled[-un_resampled_data_samples:, :series_length, :]
        data_unscaled = data_unscaled[-un_resampled_data_samples:, :series_length, :] # Since data_unscaled is the only datast where it's reasonable to keep track of the transformations.

        # Resampled data
        resampled_data = self.time_series_resampler(
                                data=data_scaled[:-un_resampled_data_samples, :, :], 
                                **resample_params
        )
        # Creating different fractions of resampled_data for training, validation and testing of the trained model later.
        samples = resampled_data.shape[0]
        train_samples = round(samples*train_samples_fraction)
        val_sample_fraction = (1 - train_samples_fraction)/2
        val_test_samples = round(samples*val_sample_fraction)
        val_sample_end = train_samples + val_test_samples

        # Training and validation data is sliced from the resampled_data dataset.
        X_train, Y_train = resampled_data[:train_samples, :, feature_start_col:feature_end_col], resampled_data[:train_samples, -1, target_col].astype(int)

        X_val, Y_val = resampled_data[train_samples:val_sample_end, :, feature_start_col:feature_end_col], resampled_data[train_samples:val_sample_end, -1, target_col].astype(int)

        return {
            'X_train': X_train,
            'Y_train': Y_train,
            'X_val': X_val,
            'Y_val': Y_val,
            'X_unresampled_unscaled': data_unscaled[:, :, feature_start_col:feature_end_col],
            'X_resample_pred': resampled_data[val_sample_end:, :, feature_start_col:feature_end_col],
            'Y_resample_pred': resampled_data[val_sample_end:, -1, target_col],
            'X_unresample_pred': un_resampled_data[:, :, feature_start_col:feature_end_col],
            'Y_unresample_pred': un_resampled_data[:, -1, 10],
        }

    def time_series_resampler(self, data: np.ndarray, target_col: int, series_length: int, class_pct: Dict[int, float]) -> np.ndarray:
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












