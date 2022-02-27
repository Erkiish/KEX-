from Strategies.Strategies import TESTRSIouStrategy
from Data_Pipelines.Sim_Data_Pipeline import DataPipeline, ScaleHandler, indicator_adder_x, rsi_adder
from ML.ML_Models import LSTM_model, CNN_LSTM_model
from ML.ML_Pipeline import MLPipeline
import numpy as np
import tensorflow as tf

# np.ndarray: numpy array with column in order: open(0), high(1), low(2), close(3), volume(4) # This is the only constant
# :::, rsi_14(5), sma_200(6),sma_30(7), pct_change(8), ema_20(9):::, # This section can vary in length and type of data, this example is based on indicator_adder_x!!
# buy_signals(10), sell_signals(11) # This changes based on what indicator_adder, that is how many indicators that are added.

# TensorBoard startup command terminal: tensorboard --logdir ML/TensorBoard_logs/fit_logs

# DataPipeline params
batch_size = 8000
time_steps = 7500
indicator_adder = rsi_adder #indicator_adder_x # for lstm # Inspect choosen function in order to have an understanding of the column structure of final array. See comment at top of file.

# Creating strategy object.
strategy = TESTRSIouStrategy(rsi_col_index=5)

# Params for and initialization of ScalerHandler instance, see example of column structure at beginning of this file if needed.
min_max_scaler_cols = [(0, 1, 2, 3), (4, )]# [(4,)] # for lstm
standardizer_scaler_cols = [] #[(8,), (0, 1, 2, 3, 6, 7, 9)] # for lstm
divider_scaler_cols = {} #{100:(5,)} # for lstm
scale_handler = ScaleHandler(
                            min_max_scaler_cols=min_max_scaler_cols, 
                            standardize_scaler_cols=standardizer_scaler_cols, 
                            divide_scaler_cols=divider_scaler_cols
) # See comment above
# Initialize DataPipeline instance.
data_pipeline = DataPipeline(
                            batch_size=batch_size, 
                            time_steps=time_steps, 
                            indicator_adder=indicator_adder,
                            strategy=strategy,
                            scale_handler=scale_handler, 
)

remove_features_from_train_val = (5,) # Because of CNN
target_col = 5 # Because of CNN
data_dict = data_pipeline.get_data_full(
    remove_features_from_train_val=remove_features_from_train_val,
    target_col=5,
    class_pct={0:0.6, 1:0.4} # Because of CNN
)

##### Model creation #####

# Fetches LSTM model with input_shape matching number of features in X_train datashape, made for binary classification.
model = CNN_LSTM_model(input_shape=(None, data_dict['X_train'].shape[2]), n_binary_classifiers=1)

##### Testing and evalutaion of the model #####
model_name = 'CNN_LSTM_test'

ML_Pipeline = MLPipeline(
    ML_model=model,
    model_name=model_name,
    data_dict=data_dict
)

EPOCHS = 50
tensorboard = True
ML_Pipeline.fit_model(
    epochs=EPOCHS,
    tensorboard=tensorboard
)

ML_Pipeline.evaluate_model(
    save_file='test_model_json_save',
    view_false_negatives=False, 
    view_false_positives=False
)
