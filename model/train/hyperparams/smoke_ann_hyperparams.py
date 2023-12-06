
from model.train.base.hyperparameters import Hyperparameters

ANN_defaults_hyperparams = {
    'TRAIN_SIZE': 0.7,
    'RANDOM_STATE': 42,
    'HIDDEN_SIZE': 30,
    'HIDDEN_SIZE_2': 10,
    'LEARNING_RATE': 0.001,
    'NUM_EPOCHS': 100,
    'BATCH_SIZE': 50
}


class Smoke_Hyperparameters(Hyperparameters):

    def __init__(self, hyperparams: dict = None, append_to_other: bool = True):
        if hyperparams is None:
            hyperparams = ANN_defaults_hyperparams
        else:
            if append_to_other:
                default_hyperparams = ANN_defaults_hyperparams
                for k, v in hyperparams.items():
                    default_hyperparams[k] = v
                hyperparams = default_hyperparams.copy()
        super().__init__(hyperparams)
