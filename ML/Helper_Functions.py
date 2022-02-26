import numpy as np
import plotly.graph_objects as go
from typing import Tuple, Union, Dict, List


def compute_result_info(y_pred: np.ndarray, y_correct: np.ndarray) -> None:

    res = y_pred == y_correct

    res_2 = y_correct - y_pred

    tot_guesses = 1
    for dim in y_pred.shape:
        tot_guesses *= dim
    
    print('Sum of all guesses: ', tot_guesses)

    sum_trade_guess = y_pred.sum()
    fp = (res_2 == -1).sum()
    fn = (res_2 == 1).sum()

    print('Sum of guesses of a trade: ', sum_trade_guess)

    print('Total number of actual trades: ', y_correct.sum())

    print('Number of correctly predicted trades: ', max(sum_trade_guess - fp - fn, 0))

    print('Number of correctly predicted non-trades: ', (y_correct == 0).sum() - (res_2 == -1).sum())

    print('Number of correct predictions: ', res.sum())

    print('Number of false positives: ', fp)

    print('Number of false negatives: ', fn)

def view_false_positives(y_pred: np.ndarray, y_correct: np.ndarray, col_y: int, n_views: int, data: np.ndarray, plot_cols: Dict[int, str]):
    """Generates plots of specified ammount of false positives and corresponding time series.

    Args:
        y_pred (np.ndarray): Prediction array.
        y_correct (np.ndarray): Array of correct classes.
        col_y (list[Union[int, str]]): Col-index of the column that the predictions x-value should be matched with.
        n_views (int): How many false positives plots to generate.
        data (np.ndarray): Data from where columns that are plotted are accessed.
        plot_cols (dict[int, str]): Dict with key as the column index and value as the column name.
    """

    view_false_pn(y_pred=y_pred, y_correct=y_correct, pn=-1, col_y=[col_y, 'False positive'], n_views=n_views, data=data, plot_cols=plot_cols)

def view_false_negatives(y_pred: np.ndarray, y_correct: np.ndarray, col_y: int, n_views: int, data: np.ndarray, plot_cols: Dict[int, str]):
    """Generates plots of specified ammount of false negatives and corresponding time series.

    Args:
        y_pred (np.ndarray): Prediction array.
        y_correct (np.ndarray): Array of correct classes.
        col_y (list[Union[int, str]]): Col-index of the column that the predictions x-value should be matched with.
        n_views (int): How many false negatives plots to generate.
        data (np.ndarray): Data from where columns that are plotted are accessed.
        plot_cols (dict[int, str]): Dict with key as the column index and value as the column name.
    """

    view_false_pn(y_pred=y_pred, y_correct=y_correct, pn=1, col_y=[col_y, 'False negative'], n_views=n_views, data=data, plot_cols=plot_cols)

def view_false_pn(y_pred: np.ndarray, y_correct: np.ndarray, pn: int, col_y: List[Union[int, str]], n_views: int, data: np.ndarray, plot_cols: Dict[int, str]):

    false_pn_index = ((y_correct - y_pred) == pn).nonzero()[0]
    nbr_false_pn = false_pn_index.shape[0]
    time_step_pos = data.shape[1] - 1
    x_index = np.linspace(0, time_step_pos, time_step_pos + 1)
    for i in range(n_views):

        if nbr_false_pn < i + 1:
            return

        fig = go.Figure()
        batch_nbr = false_pn_index[i]
        data_view = data[batch_nbr]
        y_val = data_view[-1, col_y[0]]
        fig.add_trace(go.Scatter(x=[time_step_pos], y=[y_val], mode='markers', name=col_y[1]))
        for col, name in plot_cols.items():
            fig.add_trace(go.Scatter(x=x_index, y=data_view[:, col], mode='lines', name=name))
        
        fig.show()

        


    
