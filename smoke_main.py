import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, SequentialSampler

from config.config_parser import ConfigParser
from dataset.smoke_dataset import SmokeDataset
from model.smoke_ann import SmokeNeuralNetwork
from model.train.hyperparams.smoke_ann_hyperparams import Smoke_Hyperparameters
from model.train.smoke_trainer import SmokeTrainer

RANDOM_STATE = 42


def build_dataset(cfg: ConfigParser, hyperparams: dict, batch_loader_size: int = 50) -> tuple:
    df = pd.read_csv(os.path.join(cfg.consts['DATASET_PATH'], 'smoking_train_dataset.csv'))
    # One-Hot Enc
    columns_to_onehot = []
    for col in df.columns:
        if df[col].dtype == "int64":
            if df[col].unique().__len__() < 7:
                columns_to_onehot.append(col)
    df_one_hot = pd.get_dummies(df[columns_to_onehot])
    df = df.drop(columns_to_onehot, axis=1)
    df = df.join(df_one_hot)
    # Standardization
    columns_to_normalize = []
    for col in df.columns:
        if df[col].dtype == "float64":
            columns_to_normalize.append(col)
    scaler = StandardScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    # Split dataset into train and test
    X = df.drop(columns=["smoking"]).values
    y = df["smoking"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=RANDOM_STATE)
    # Convert into my dataset cutom class
    train_dataset = SmokeDataset(X_train, y_train)
    test_dataset = SmokeDataset(X_test, y_test)
    # Use data-loader in order to have batches
    train_loader = DataLoader(train_dataset, batch_size=batch_loader_size, shuffle=False, num_workers=0,
                              sampler=SequentialSampler(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=batch_loader_size, shuffle=False, num_workers=0,
                             sampler=SequentialSampler(test_dataset))
    return train_loader, test_loader


def main():
    # Get project configurations
    cfg = ConfigParser()
    hyperparams = Smoke_Hyperparameters().hyperparameters
    # Prepare dataset
    train_loader, test_loader = build_dataset(cfg, hyperparams, batch_loader_size=hyperparams['BATCH_SIZE'])
    # Instantiate the model
    model = SmokeNeuralNetwork(train_loader.dataset.X.shape[1], 1, hyperparams['HIDDEN_SIZE'],
                               hyperparams['HIDDEN_SIZE_2'])
    trainer = SmokeTrainer(model, name='Smoke-ANN')
    train_losses, test_losses = trainer.train_loader(train_loader, test_loader)
    # Plot the train loss and test loss per iteration
    fig = trainer.draw_train_test_loss(train_losses, test_losses)
    trainer.save_image('Smoke ANN - Train and test loss', fig)


if __name__ == "__main__":
    main()
