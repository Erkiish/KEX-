if __name__ == '__main__':
    from Technical_Indicators import *
else:
    from Strategies.Technical_Indicators import *

import pandas as pd
import numpy as np
from dataclasses import dataclass
from abc import ABCMeta, abstractmethod

class Strategy(metaclass=ABCMeta):

    @abstractmethod
    def get_buy_signals(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_sell_signals(self, data: np.ndarray) -> np.ndarray:
        pass


@dataclass
class TESTRSIouStrategy(Strategy):

    rsi_col_index: int
    rsi_entry_cross: int=30
    rsi_exit_cross: int=55

    def get_buy_signals(self, data: np.ndarray) -> np.ndarray:
        """For now only returns buy-signals. The buy signal time-series is shifted forward 1 day so that it can be interpreted as a buy from
        a signal that showed the day before."""
        rsi = data[:, self.rsi_col_index]
        data = data[1:]
        rsi_before = rsi[:-1]
        rsi_after = rsi[1:]
        rsi_below = rsi_before < self.rsi_entry_cross
        rsi_over = rsi_after > self.rsi_entry_cross
        buy_data = np.concatenate(([0], np.logical_and(rsi_below, rsi_over)))[:-1]
        return np.hstack([data, buy_data.reshape(-1, 1)])
    
    def get_sell_signals(self, data: np.ndarray) -> np.ndarray:
        """Same logic right as for get_buy_signal method doc."""
        rsi = data[:, self.rsi_col_index]
        data = data[1:]
        rsi_before = rsi[:-1]
        rsi_after = rsi[1:]
        rsi_below = rsi_before > self.rsi_exit_cross
        rsi_over = rsi_after < self.rsi_exit_cross
        sell_data = np.concatenate(([0], np.logical_and(rsi_below, rsi_over)))[:-1]
        return np.hstack([data, sell_data.reshape(-1, 1)])

