import tensorflow as tf
import keras_tuner as kt
from typing import Tuple, Callable
import os
from ML.CONSTANSTS import HP_TUNER_RESULT_PATH

# HPT = Hyper Parameter Tuning, this module uses the keras_tuner library for the task of finding optimal hyper parameters.

# For info and citation: https://keras.io/guides/keras_tuner/getting_started/

def build_LSTM_model(hp: kt.HyperParameters) -> tf.keras.Model:

    model = tf.keras.Sequential()
    # Input layer has separate inputs
    model.add(tf.keras.layers.LSTM(units=hp.Int(name='Input units',min_value=5, max_value=75, step=5), return_sequences=True, input_shape=(None, 12)))
    
    # could introduce a HyperParameter for kind of layer, for example have a line with: 
    # layer = hp.Choice('Layer type', [tf.keras.layers.LSTM, tf.keras.layers.GRU, tf.keras.layers.simpleRNN])

    # Testing for optimal ammount of layers.
    for layer in range(hp.Int(name='Number of layers', min_value=1, max_value=10, step=1)):

        model.add(tf.keras.layers.LSTM(
                                        units=hp.Int(name=f'Units_layer_{layer}', min_value=8, max_value=100, step=5),
                                        return_sequences=True
                                        )
                
                 ) # Could add activation function HyperParameter.
        # Could also add dropout HyperParameters!!
    # Last LSTM layer has return_sequences=False for many-to-one
    model.add(tf.keras.layers.LSTM(units=hp.Int(name='Units_layer_last_LSTM', min_value=2, max_value=25, step=5), return_sequences=False))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    # Learning rate:
    learning_rate = hp.Float('Learning rate', min_value=1e-4, max_value=1e-1, step=0.05)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )
    print(model.summary())
    return model

print(os.getcwd())

def HPTBayesian(hypermodel: Callable, project_name: str, max_trials: int=1000):
    
    

    kt.BayesianOptimization(
        hypermodel=hypermodel,
        objective='val_accuracy',
        max_trials=max_trials,
        directory=HP_TUNER_RESULT_PATH,
        project_name=project_name
    )

