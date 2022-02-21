if __name__ == '__main__':
    from Technical_Indicators import *
else:
    from Strategies.Technical_Indicators import *

import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class TESTRSIouStrategy:

    rsi_interval: int=14
    rsi_entry_cross: int=30
    rsi_exit_cross: int=55

    def get_buy_signals(self, data: pd.DataFrame) -> pd.Series:
        """For now only returns buy-signals. The buy signal time-series is shifted forward 1 day so that it can be interpreted as a buy from
        a signal that showed the day before."""
        rsi = data[f'rsi_{self.rsi_interval}']
        data = data.iloc[1:]
        rsi_before = rsi.to_numpy()[:-1]
        rsi_after = rsi.to_numpy()[1:]
        rsi_below = rsi_before < self.rsi_entry_cross
        rsi_over = rsi_after > self.rsi_entry_cross
        data['buy_signal'] = np.concatenate(([0], np.logical_and(rsi_below, rsi_over)))[:-1]
        return data
    
    def get_sell_signals(self, data: pd.DataFrame) -> pd.Series:
        """Same logic right as for get_buy_signal method doc."""
        rsi = data[f'rsi_{self.rsi_interval}']
        data = data.iloc[1:]
        rsi_before = rsi.to_numpy()[:-1]
        rsi_after = rsi.to_numpy()[1:]
        rsi_below = rsi_before > self.rsi_exit_cross
        rsi_over = rsi_after < self.rsi_exit_cross
        data['sell_signal'] = np.concatenate(([0], np.logical_and(rsi_below, rsi_over)))[:-1]
        return data

