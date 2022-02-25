from random import sample
from Strategies.Strategies import TESTRSIouStrategy
from Data_Pipelines.Sim_Data_Pipeline import test_pipeline_x, sim_data_getter_x, time_series_resampler, ScalerHandler
import pandas as pd
import plotly.graph_objects as go
from ML_Models.Pure_LSTM import LSTM_model_x_creator
import numpy as np


data, col_map = test_pipeline_x(6500, 6000, TESTRSIouStrategy(), ScalerHandler([('volume',)], [('pct_change',)], {100:('rsi_14',), 'MAX':('close', 'open', 'low', 'high', 'ema_20', 'sma_30', 'sma_200')}))

print(data.shape)
nbr_og_data_samples = 3000
data_1 = data[-nbr_og_data_samples:, :, :]

data = time_series_resampler(data[:-nbr_og_data_samples, :, :], target_col=10, series_length=30, class_pct={0:0.6, 1:0.4})

print(data.shape)

feature = col_map['rsi_14']

y_col = col_map['buy_signal']

samples = data.shape[0]

train_samples = round(samples*0.6)
val_test_samples = round(samples*0.2)
val_sample_end = train_samples + val_test_samples

X_train, Y_train = data[:train_samples, :, :10], data[:train_samples, -1, 10].astype(int)

X_val, Y_val = data[train_samples:val_sample_end, :, :10], data[train_samples:val_sample_end, -1, 10].astype(int)

def compute_sample_weight(Y_train: np.ndarray):
    sample_weight = np.ones(shape=Y_train.shape)
    sample_weight[Y_train == 1] == 10
    sample_weight[Y_train == 0] == 0.1


model = LSTM_model_x_creator(input_shape=(None, X_train.shape[2]), n_binary_classifiers=1, return_sequences=False)#X_train.shape[2])


model.fit(X_train, Y_train, epochs=65, validation_data=(X_val, Y_val))

# NEJ, DENNA METODEN SUGER BALLE. DET GÄLLER ATT BYGGA OM FUNKTIONER SOM HANTERAR VIKTERNA.. TYP LOSS-FUNKTIONEN, SE: https://github.com/keras-team/keras/issues/2115
def train_model(model):

    for epoch in range(20):
        for batch in range(X_train.shape[0]):
            model.fit(X_train[:, batch, :, :], Y_train[: ,batch, :], validation_data=(X_val[0], Y_val[0]), epochs=epoch+1, initial_epoch=epoch, class_weight={0:0.01, 1:2})

    return model

def compute_result_info(y_pred: np.ndarray, y_correct: np.ndarray) -> None:
    res = y_pred == y_correct

    res_2 = y_correct - y_pred

    print('Sum of all guesses: ', y_pred.shape[0])

    print('Sum of guesses of a trade: ', y_pred.sum())

    print('Total number of actual trades: ', y_correct.sum())

    print('Number of correctly predicted values: ', res.sum())

    print('Number of false positives: ', (res_2 == -1).sum())

    print('Number of false negatives: ', (res_2 == 1).sum())



y_correct = data[val_sample_end:, -1, 10]
print(y_correct.shape)

print(data[val_sample_end:, :, :10].shape)
y_pred = np.rint(model.predict(data[val_sample_end:, :, :10]))[:, 0]
print(y_pred.shape)

compute_result_info(y_pred, y_correct)


y_correct = data_1[:, -1, 10]
print(y_correct.shape)

print(data_1[:, :, :10].shape)
y_pred = np.rint(model.predict(data_1[:, :, :10]))[:, 0]

compute_result_info(y_pred, y_correct)






