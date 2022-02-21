import pandas_ta as ta
import pandas as pd

def add_rsi(data: pd.DataFrame, len: int=14) -> pd.DataFrame:

    data[f'rsi_{len}'] = ta.rsi(data['close'], len)

    return data

def add_sma(data: pd.DataFrame, len: int=50) -> pd.DataFrame:

    data[f'sma_{len}'] = ta.sma(data['close'], len)

    return data

def add_ema(data: pd.DataFrame, len: int=30) -> pd.DataFrame: 

    data[f'ema_{len}'] = ta.ema(data['close'], len)

    return data

def add_obv(data: pd.DataFrame) -> pd.DataFrame:

    data['obv'] = ta.obv(data['close'], data['volume'])

    return data

def add_ad(data: pd.DataFrame) -> pd.DataFrame:

    data['ad'] = ta.ad(data['high'], data['low'],data['close'], data['volume'])

    return data 

def add_adx(data: pd.DataFrame) -> pd.DataFrame:

    data['adx'] = ta.adx(data['high'], data['low'], data['close'], length=14)

    return data


