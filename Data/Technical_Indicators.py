import pandas_ta as ta
import pandas as pd


def add_rsi(data: pd.DataFrame, len: int=14):

    data[f'rsi_{len}'] = ta.rsi(data['close'], len)

    return data

