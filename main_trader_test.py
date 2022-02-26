from Strategies.Strategies import TESTRSIouStrategy
from Data_Pipelines.Sim_Data_Pipeline import DataPipeline, ScaleHandler, indicator_adder_x
import plotly.graph_objects as go
from ML.Pure_LSTM import LSTM_model_x_creator
from ML.Helper_Functions import compute_result_info
import numpy as np

# np.ndarray: numpy array with column in order: open(0), high(1), low(2), close(3), volume(4) # This is the only constant
# :::, rsi_14(5), sma_200(6),sma_30(7), pct_change(8), ema_20(9):::, # This section can vary in length and type of data, this example is based on indicator_adder_x!!
# buy_signals(10), sell_signals(11) # This changes based on what indicator_adder, that is how many indicators that are added.

# DataPipeline params
batch_size = 2000
time_steps = 1500
indicator_adder = indicator_adder_x # Inspect choosen function in order to have an understanding of the column structure of final array. See comment at top of file.
strategy = TESTRSIouStrategy(rsi_col_index=5)
# Params for and initialization of ScalerHandler instance, see example of column structure at beginning of this file if needed.
min_max_scaler_cols = [(4,)]
standardizer_scaler_cols = [(8,), (0, 1, 2, 3, 6, 7, 9)]
divider_scaler_cols = {100:(5,)}
scale_handler = ScaleHandler(
                            min_max_scaler_cols=min_max_scaler_cols, 
                            standardizer_scaler_cols=standardizer_scaler_cols, 
                            divider_scaler_cols=divider_scaler_cols
) # See comment above
# Initialize DataPipeline instance.
data_pipeline = DataPipeline(
                            batch_size=batch_size, 
                            time_steps=time_steps, 
                            indicator_adder=indicator_adder,
                            strategy=strategy,
                            scale_handler=scale_handler
)

# Gets data which has not been resampled
data = data_pipeline.get_data()

# Reserves a fraction of data that has not been resampled for evaluation of model later.
un_resampled_data_samples = round(batch_size*1/3)
un_resampled_data = data[-un_resampled_data_samples:, :20, :]

# Parameters for resampling of fraction of data                  
resample_params = {
                    'target_col': 10,
                    'series_length': 30, 
                    'class_pct': {0:0.9, 1:0.1}
}

# Resampled data
resampled_data = data_pipeline.time_series_resampler(
                                                    data=data[:-un_resampled_data_samples, :, :], 
                                                    **resample_params
)

# Printing shape of ndarray for better introspection in data.
print(resampled_data.shape)

##### Model data preparation #####

# Creating different fractions of resampled_data for training, validation and testing of the trained model later.
samples = resampled_data.shape[0]
train_samples = round(samples*0.6)
val_test_samples = round(samples*0.2)
val_sample_end = train_samples + val_test_samples

# Training and validation data is sliced from the resampled_data dataset.
X_train, Y_train = resampled_data[:train_samples, :, :10], resampled_data[:train_samples, -1, 10].astype(int)

X_val, Y_val = resampled_data[train_samples:val_sample_end, :, :10], resampled_data[train_samples:val_sample_end, -1, 10].astype(int)

##### Model creation #####

# Fetches LSTM model with input_shape matching number of features in X_train datashape, made for binary classification.
model = LSTM_model_x_creator(input_shape=(None, X_train.shape[2]), n_binary_classifiers=1, return_sequences=False)

##### Testing of the model #####
epochs = 65
model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_val, Y_val))


##### Testing of the model #####

# Testing with parts of the resampled_data
print('Resampled data test metrics: ')
y_correct = resampled_data[val_sample_end:, -1, 10]
y_pred = np.rint(model.predict(resampled_data[val_sample_end:, :, :10]))[:, 0]
compute_result_info(y_pred, y_correct)

print('Original data (not resampled) test metrics: ')
y_correct = un_resampled_data[:, -1, 10]
y_pred = np.rint(model.predict(un_resampled_data[:, :, :10]))[:, 0]
compute_result_info(y_pred, y_correct)



# Code below may be used later, but for now is unnecessary

# Function for creating sample_weights, not currently needed
def compute_sample_weight(Y_train: np.ndarray):
    sample_weight = np.ones(shape=Y_train.shape)
    sample_weight[Y_train == 1] == 10
    sample_weight[Y_train == 0] == 0.1



# NEJ, DENNA METODEN SUGER BALLE. DET GÄLLER ATT BYGGA OM FUNKTIONER SOM HANTERAR VIKTERNA.. TYP LOSS-FUNKTIONEN, SE: https://github.com/keras-team/keras/issues/2115
def train_model(model):

    for epoch in range(20):
        for batch in range(X_train.shape[0]):
            model.fit(X_train[:, batch, :, :], Y_train[: ,batch, :], validation_data=(X_val[0], Y_val[0]), epochs=epoch+1, initial_epoch=epoch, class_weight={0:0.01, 1:2})

    return model