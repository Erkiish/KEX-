import tensorflow as tf

def LSTM_model(input_shape: int, n_binary_classifiers: int=1) -> tf.keras.Model:

    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(20, return_sequences=True),
        tf.keras.layers.LSTM(20, return_sequences=True),
        tf.keras.layers.LSTM(10, return_sequences=True),
        tf.keras.layers.LSTM(5, return_sequences=False),
        tf.keras.layers.Dense(n_binary_classifiers, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.BinaryAccuracy()])
    print(model.summary())
    return model

def CNN_LSTM_model(input_shape: int, n_binary_classifiers: int=1) -> tf.keras.Model:

    # USE KERAS FUNCTIONAL API FOR MULTINPUT NETWORKS => DIFFERENT TYPES OF CONV1D LAYERS CAN FEED ON RAW-DATA INSTEAD OF FROM EACHOTHERS ALREADY TRANSFORMED DATA.

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=40, kernel_size=10, input_shape=input_shape),
        tf.keras.layers.Conv1D(filters=30, kernel_size=5),
        tf.keras.layers.Conv1D(filters=20, kernel_size=2),
        tf.keras.layers.LSTM(20, return_sequences=True),
        tf.keras.layers.LSTM(20, return_sequences=True),
        tf.keras.layers.LSTM(10, return_sequences=True),
        tf.keras.layers.LSTM(5, return_sequences=False),
        tf.keras.layers.Dense(n_binary_classifiers, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.BinaryAccuracy()])
    print(model.summary())
    return model