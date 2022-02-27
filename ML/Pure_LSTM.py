import tensorflow as tf

def LSTM_model_x_creator(input_shape: int, n_binary_classifiers: int=1, return_sequences: bool=False) -> tf.keras.Model:

    model_x = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(20, return_sequences=True),
        tf.keras.layers.LSTM(20, return_sequences=True),
        tf.keras.layers.LSTM(10, return_sequences=True),
        tf.keras.layers.LSTM(5, return_sequences=return_sequences),
        tf.keras.layers.Dense(n_binary_classifiers, activation='sigmoid'),
    ])
    model_x.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.BinaryAccuracy()])
    print(model_x.summary())
    return model_x
