import pandas as pd 

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


