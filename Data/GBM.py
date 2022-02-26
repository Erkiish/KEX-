import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Union, Dict
from scipy.stats import norm, uniform
import random
import plotly.graph_objects as go
try:
    from Full_Data import FullData
except:
    from Data.Full_Data import FullData

@dataclass
class GBM:

    init_value: int=100
    drift: float=0.1
    std: float=0.3  

    def generate_data(self, length: int=250, type: Union[str, bool]=False) -> np.ndarray:

        time_end = length/252

        if type == 'ohlc':
            return self.generate_ohlcv_data(length)
        
        time_array = np.linspace(0, time_end, num=length)
        dt = time_array.iloc[1, 0] - time_array.iloc[0, 0]

        wiener_process = self._generate_wiener_process(self.init_value, dt, length)
        return self._generate_GBM_process(wiener_process, time_array)
   
    def generate_ohlcv_data(self, length: int=250, intra_mean: float=0.006, intra_std: float=0.005) -> np.ndarray:
        """Generates ohlcv data (open, high, low, close, volume) in np.ndarray form.
        The columns correspond to the following dict: {'open':0, 'high':1, 'low':2, 'close':3, 'volume':4}

        Args:
            length (int, optional): _description_. Defaults to 250.
            intra_mean (float, optional): _description_. Defaults to 0.006.
            intra_std (float, optional): _description_. Defaults to 0.005.

        Returns:
            pd.DataFrame: _description_
        """

        time_end = length/1008

        time_array = np.linspace([0], [time_end], num=length)
        self.dt = time_array[1, 0] - time_array[0, 0]

        start_values = [self.init_value]

        wiener_process = self._generate_wiener_process(start_values, self.dt, length).T
        raw_gbm_data = self._generate_GBM_process(wiener_process, time_array)#.rename(columns={0:'open', 1:'low', 2:'high', 3:'close'})

        raw_gbm_data = np.stack([raw_gbm_data, raw_gbm_data, raw_gbm_data, raw_gbm_data], axis=-1)

        intra_day_var = norm.rvs(size=(length, 4), loc=intra_mean, scale=intra_std) + 1

        close_col = round(uniform.rvs(loc=0, scale=3))
        cols = [0, 1, 2, 3]
        cols.remove(close_col)
        open_col = cols[round(uniform.rvs(loc=0, scale=2))]

        ohlc_raw = intra_day_var*raw_gbm_data

        subtract_drift = time_array[:, 0] 
        subtract_drift[:] = self.drift/(len(subtract_drift)*10) + 1
        subtract_drift = np.cumprod(subtract_drift)*self.init_value
        volume = subtract_drift*norm.rvs(size=(length,), loc=1000, scale=30)

        # ['open', 'high', 'low', 'close', 'volume']
        return np.stack([ohlc_raw[:, open_col], ohlc_raw.max(axis=1), ohlc_raw.min(axis=1), ohlc_raw[:, close_col], volume], 
                                axis=-1)

    def _generate_wiener_process(self, start_value: Union[float, list], dt: float, n_steps: int) -> np.ndarray:

        start_values = np.asarray(start_value)
        return norm.rvs(size=start_values.shape + (n_steps,), scale=1)
    
    def _generate_GBM_process(self, wiener_process: np.ndarray, time_array: np.ndarray, std: Union[float, bool]=False, drift: Union[float, bool]=False) -> np.ndarray:

        if not std:
            std = self.std
        if not drift:
            drift = self.drift
        return self.init_value*np.cumprod(np.exp(std*wiener_process*np.sqrt(self.dt) + (drift - (std**2)/2)*self.dt))#(drift - (std**2)/2)


class ParameterEstimation:

    def __init__(self, user: Union[str, bool]=False, data: Union[Dict[str, pd.DataFrame], bool]=False):
        
        if user:
            self.data_class = FullData(user)

            self.data = self.data_class.get_data(start_date='1998-01-01')
        else:
            self.data = data
    
    def estimate_drift(self) -> float:

        self.drift = 0
        for df in self.data.values():

            self.drift += ((df.loc[:, 'close'].iloc[1:].to_numpy() - df.loc[:, 'close'].iloc[:-1].to_numpy())/df.loc[:, 'close'].iloc[:-1].to_numpy()).sum()/(len(df) - 1)

        self.drift = self.drift/len(self.data)*252

        return self.drift

    def estimate_std(self) -> float:

        self.std = 0
        for df in self.data.values():

            self.std += np.sqrt((((df.loc[:, 'close'].iloc[1:].to_numpy() - df.loc[:, 'close'].iloc[:-1].to_numpy())/df.loc[:, 'close'].iloc[:-1].to_numpy() - self.drift)**2).sum()/(len(df) - 1))

        self.std = self.std/len(self.data)

        return self.std*3

    def estimate_intraday_dist(self) -> dict:

        self.intra_mean = 0
        for df in self.data.values():
            
            self.intra_mean += ((df.loc[:, 'high'].to_numpy() - df.loc[:, 'low'].to_numpy())/df.loc[:, 'low'].to_numpy()).sum()/(len(df) - 1)

        self.intra_mean = self.intra_mean/len(self.data)

        self.intra_std = 0
        for df in self.data.values():

            self.intra_std += np.sqrt((((df.loc[:, 'high'].to_numpy() - df.loc[:, 'low'].to_numpy())/df.loc[:, 'low'].to_numpy() - self.intra_mean)**2).sum()/(len(df) - 1))
        
        self.intra_std = self.intra_std/len(self.data)

        return self.intra_mean, self.intra_std
    







if __name__ == '__main__':
    from Nasdaq_API import nasdaq_get_history
    from Data_Cleaning import nan_handler, zero_handler
    test = zero_handler(nan_handler(nasdaq_get_history('SE0000337842', '1998-01-01', '2022-01-01')))
    data = {'omx':test}
    a = ParameterEstimation(data=data)
    drift = a.estimate_drift()
    print(drift)
    std = a.estimate_std()
    print(std)
    intra_mean, intra_std = a.estimate_intraday_dist()

    print(intra_mean, intra_std)

    x = True
    if x:
        test_x = GBM(drift=drift, std=std)
        
        test = nasdaq_get_history('SE0000337842', '2020-01-01', '2022-01-01')
        data = test_x.generate_ohlcv_data(length=504)
        go.Figure(data=go.Scatter(x=data.index, y=data['close'])).show()
        go.Figure(data=go.Ohlc(low=data['low'], high=data['high'], close=data['close'], open=data['open'], x=data.index)).show()
        #go.Figure(data=go.Ohlc(x=data.index, open=data['open'], close=data['close'], low=data['low'], high=data['high'])).show()
        go.Figure(data=go.Ohlc(x=test.index, open=test['open'], close=test['close'], low=test['low'], high=test['high'])).show()