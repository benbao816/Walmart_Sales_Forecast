from src.entities.lstm_params import LSTMParams
from keras.models import Sequential
from keras.layers import LSTM, Dense, LayerNormalization
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf

def build_lstm(lstm_param:LSTMParams,
               features,
               target,
               shape) -> Sequential:
    model = Sequential()

    model.add(LSTM(units=lstm_param.lstm_units,
                   return_sequences=True, 
                   input_shape=(shape[1],shape[2]),
                   dropout=lstm_param.dropout,
                   recurrent_dropout=lstm_param.recurrent_dropout))
    
    model.add(LayerNormalization(axis=lstm_param.layer_norm_axis,
                                 beta_initializer=lstm_param.beta_initializer,
                                 gamma_initializer=lstm_param.gamma_initializer))

    model.add(LSTM(units=lstm_param.lstm_units,
                   dropout=lstm_param.dropout,
                   recurrent_dropout=lstm_param.recurrent_dropout))
    
    model.add(LayerNormalization(axis=lstm_param.layer_norm_axis,
                                 beta_initializer=lstm_param.beta_initializer,
                                 gamma_initializer=lstm_param.gamma_initializer))
    
    model.add(Dense(units=lstm_param.dense_unit))
    
    opt = tf.optimizers.Adam(learning_rate=lstm_param.learning_rate)
    model.compile(optimizer=opt, loss='mean_squared_error')
    # model.compile(optimizer=lstm_param.optimizer, loss=lstm_param.loss)
    model.fit(features,target, epochs=lstm_param.epochs, batch_size=lstm_param.batch_size, verbose=1)
    return model