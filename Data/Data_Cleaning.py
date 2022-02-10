import pandas as pd
import numpy as np

def time_frame_getter(data: pd.DataFrame, time_frame: str) -> pd.DataFrame:
    """
    Function for turning 'daily' datasets into either 'weekly' or 'monthly'.

    Args:
        data (pd.DataFrame): pandas DataFrame that has column 'close'
        time_frame (str): Either 'W-FRI' for weekly or 'M' for monthly

    Returns:
        pd.DataFrame: Returns pandas DataFrame with the specified time_frame.
    """

    return data.resample(time_frame).ohlc()


def nan_handler(df: pd.DataFrame) -> pd.DataFrame:
    """Handles nan values in a price-dataframe. Takes average, closing price, forward fill and lastly backward fill if necessary

    Args:
        df (pd.DataFrame): dataframe that has columns 'close', 'high', 'low', 'open'

    Returns:
        [pd.DataFrame]: dataframe without nan values
    """
    bool = (df['high'].notna()) & (df['open'].isna()) & (df['low'].notna())
    df.loc[bool, 'open'] = (df.loc[bool, 'high'] + df.loc[bool, 'low'])/2
    for label in ['open', 'low', 'high']:
        df.loc[(df[f'{label}'].isna()) & (df['close'].notna()), f'{label}'] = df.loc[(df[f'{label}'].isna()) & (df['close'].notna()), 'close']
    return df.fillna(method='ffill').fillna(method='bfill')

def zero_handler(df: pd.DataFrame) -> pd.DataFrame:
    """Handles zero-values that can be interpreted as nan-values in a price-dataframe.

    Args:
        df (pd.DataFrame): dataframe that has columns 'close', 'high', 'low', 'open'

    Returns:
        [pd.DataFrame]: df that does not have any zero-values where it should not.
    """
    bool = (df['high'] != 0) & (df['open'] == 0) & (df['low'] != 0)
    df.loc[bool, 'open'] = (df.loc[bool, 'high'] + df.loc[bool, 'low'])/2
    for label in ['open', 'low', 'high']:
        df.loc[(df[f'{label}'] == 0) & (df['close'] != 0), f'{label}'] = df.loc[(df[f'{label}'] == 0) & (df['close'] != 0), 'close']
        if (df[f'{label}'] == 0).sum() > 0:
            df.loc[(df[f'{label}'] == 0), f'{label}'] = df.loc[(df[f'{label}'] == 0), f'{label}'].replace(0, np.nan).fillna(method='ffill').fillna(method='bfill')
    return df