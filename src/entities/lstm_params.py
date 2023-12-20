from dataclasses import dataclass, field


# @dataclass
# class ModelParams:
#     model_type: str

@dataclass
class LSTMParams:
    time_step: int
    lstm_units: int
    dense_unit: int
    model_type: str = field(default="LSTM")
    optimizer: str = field(default='adam')
    loss: str = field(default='mean_squared_error')
    train_rate: float = field(default=0.8)
    dropout: float = field(default=0.1)
    recurrent_dropout: float = field(default=0.1)
    layer_norm_axis: int = field(default=1)
    beta_initializer: str = field(default="ones")
    gamma_initializer: str = field(default="zeros")
    learning_rate: float = field(default=0.0015)
    epochs: int = field(default=2)
    batch_size: int = field(default=1)


