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
        """For now only returns buy-signals. That is, the earliest time for an actual buy would be at opening the day after the signal."""
        rsi = data[f'rsi_{self.rsi_interval}']
        data = data.iloc[1:]
        rsi_before = rsi.to_numpy()[:-1]
        rsi_after = rsi.to_numpy()[1:]
        rsi_below = rsi_before < self.rsi_entry_cross
        rsi_over = rsi_after > self.rsi_entry_cross
        return data.loc[np.logical_and(rsi_below, rsi_over)[:-1], :]
    
    def get_sell_signals(self, data: pd.DataFrame) -> pd.Series:
        """Same logic right now as for get_buy_signal method doc."""
        rsi = data[f'rsi_{self.rsi_interval}']
        data = data.iloc[1:]
        rsi_before = rsi.to_numpy()[:-1]
        rsi_after = rsi.to_numpy()[1:]
        rsi_below = rsi_before > self.rsi_exit_cross
        rsi_over = rsi_after < self.rsi_exit_cross
        return data.loc[np.logical_and(rsi_below, rsi_over)[:-1], :]

