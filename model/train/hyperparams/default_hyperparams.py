import enum

from model.train.base.hyperparameters import Hyperparameters


_Defaults = {
    'LEARNING_RATE': 0.001,
    'NUM_EPOCHS': 75
}


class DefaultHyperparameters(Hyperparameters):

    def __init__(self):
        super().__init__(_Defaults)
