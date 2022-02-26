import numpy as np

def compute_result_info(y_pred: np.ndarray, y_correct: np.ndarray) -> None:

    res = y_pred == y_correct

    res_2 = y_correct - y_pred

    tot_guesses = 1
    for dim in y_pred.shape:
        tot_guesses *= dim
    
    print('Sum of all guesses: ', tot_guesses)

    print('Sum of guesses of a trade: ', y_pred.sum())

    print('Total number of actual trades: ', y_correct.sum())

    print('Number of correctly predicted trades: ', res.sum())

    print('Number of correctly predicted non-trades: ', (y_correct == 0).sum() - (res_2 == -1).sum())

    print('Number of correctly predicted classes: ', res.sum())

    print('Number of false positives: ', (res_2 == -1).sum())

    print('Number of false negatives: ', (res_2 == 1).sum())