from random import sample
from Strategies.Strategies import TESTRSIouStrategy
from Data_Pipelines.Sim_Data_Pipeline import test_pipeline_x, sim_data_getter_x, ScalerHandler
import pandas as pd
import plotly.graph_objects as go
from ML_Models.Pure_LSTM import LSTM_model_x_creator
import numpy as np


data, col_map = test_pipeline_x(300, 150, TESTRSIouStrategy(), ScalerHandler([('volume',)], [('pct_change',), ('close', 'open', 'low', 'high', 'ema_20', 'sma_30', 'sma_200')], {100:('rsi_14',)}))


feature = col_map['rsi_14']

y_col = col_map['buy_signal']



X_train, Y_train = data[:, :70, :, :10], data[:, :70, :, 10].astype(int)

X_val, Y_val = data[:, 70:100, :, :10], data[:, 70:100, :, 10].astype(int)

sample_weight = np.ones(shape=Y_train.shape)
sample_weight[Y_train == 1] == 10
sample_weight[Y_train == 0] == 0.1

print((X_val.shape[2], X_train.shape[3]))

model = LSTM_model_x_creator(input_shape=(X_val.shape[2], X_train.shape[3]), n_binary_classifiers=1)#X_train.shape[2])


# model.fit(X_train, Y_train, epochs=2, validation_data=(X_val, Y_val), sample_weight=sample_weight)


print(X_train[0, :, :].shape)


# NEJ, DENNA METODEN SUGER BALLE. DET GÄLLER ATT BYGGA OM FUNKTIONER SOM HANTERAR VIKTERNA.. TYP LOSS-FUNKTIONEN, SE: https://github.com/keras-team/keras/issues/2115
def train_model(model):

    for epoch in range(20):
        for batch in range(X_train.shape[0]):
            model.fit(X_train[:, batch, :, :], Y_train[: ,batch, :], validation_data=(X_val[0], Y_val[0]), epochs=epoch+1, initial_epoch=epoch, class_weight={0:0.01, 1:2})

    return model


model = train_model(model)

y_correct = data[0, 100:150, :, 10]

y_pred = np.rint(model.predict(data[0, 100:150, :, :10]))[:,:, 0]

res = y_pred == y_correct

res_2 = y_correct - y_pred

print('Sum of all guesses: ', y_pred.shape[0]*y_pred.shape[1])

print('Sum of guesses of a trade: ', y_pred.sum())

print('Total number of actual trades: ', y_correct.sum())

print('Number of correctly predicted values: ', res.sum())

print('Number of false positives: ', (res_2 == -1).sum())

print('Number of false negatives: ', (res_2 == 1).sum())



