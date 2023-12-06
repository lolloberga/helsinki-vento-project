import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, SequentialSampler

from config.config_parser import ConfigParser
from dataset.stroke_dataset import StrokeDataset
from model.stroke_ann import StrokeNeuralNetwork
from model.train.hyperparams.stroke_ann_hyperparams import Stroke_Hyperparameters
from model.train.stroke_trainer import StrokeTrainer

RANDOM_STATE = 42


def build_dataset(cfg: ConfigParser, hyperparams: dict, batch_loader_size: int = 50) -> tuple:
    df = pd.read_csv(os.path.join(cfg.consts['DATASET_PATH'], 'healthcare-dataset-stroke-data.csv'))
    df = df.drop(columns=['id'])
    # One-Hot Enc
    columns_to_onehot = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type',
                         'smoking_status']
    df_one_hot = pd.get_dummies(df[columns_to_onehot])
    df = df.drop(columns_to_onehot, axis=1)
    df = df.join(df_one_hot)
    # Standardization
    columns_to_normalize = ['age', 'avg_glucose_level', 'bmi']
    scaler = StandardScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    # Remove NaN rows from BMI features
    mean_stroke_1 = df[df['stroke'] == 1]['bmi'].dropna().mean()
    mean_stroke_0 = df[df['stroke'] == 0]['bmi'].dropna().mean()
    df.loc[df['stroke'] == 1, 'bmi'] = df[df['stroke'] == 1]['bmi'].fillna(mean_stroke_1)
    df.loc[df['stroke'] == 0, 'bmi'] = df[df['stroke'] == 0]['bmi'].fillna(mean_stroke_0)
    # Split dataset into train and test
    X = df.drop(columns=["stroke"]).values
    y = df["stroke"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=RANDOM_STATE)
    # Convert into my dataset cutom class
    train_dataset = StrokeDataset(X_train, y_train)
    test_dataset = StrokeDataset(X_test, y_test)
    # Use data-loader in order to have batches
    train_loader = DataLoader(train_dataset, batch_size=batch_loader_size, shuffle=False, num_workers=0,
                              sampler=SequentialSampler(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=batch_loader_size, shuffle=False, num_workers=0,
                             sampler=SequentialSampler(test_dataset))
    return train_loader, test_loader


def main():
    # Get project configurations
    cfg = ConfigParser()
    hyperparams = Stroke_Hyperparameters().hyperparameters
    # Prepare dataset
    train_loader, test_loader = build_dataset(cfg, hyperparams, batch_loader_size=hyperparams['BATCH_SIZE'])
    # Instantiate the model
    model = StrokeNeuralNetwork(train_loader.dataset.X.shape[1], 1, hyperparams['HIDDEN_SIZE'], hyperparams['HIDDEN_SIZE_2'])
    trainer = StrokeTrainer(model, name='Stroke-ANN')
    train_losses, test_losses = trainer.train_loader(train_loader, test_loader)
    # Plot the train loss and test loss per iteration
    fig = trainer.draw_train_test_loss(train_losses, test_losses)
    trainer.save_image('Stroke ANN - Train and test loss', fig)


if __name__ == "__main__":
    main()
