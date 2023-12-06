from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.train.base.hyperparameters import Hyperparameters
from model.train.base.trainer import Trainer
from model.train.hyperparams.stroke_ann_hyperparams import Stroke_Hyperparameters

THRESHOLD = 0.5


class StrokeTrainer(Trainer):

    def __init__(self, model: nn.Module, name: str = None, criterion=None, optimizer: torch.optim.Optimizer = None,
                 writer: SummaryWriter = None, hyperparameters: Hyperparameters = None) -> None:
        self._name = name
        if hyperparameters is None:
            hyperparameters = Stroke_Hyperparameters()

        super().__init__(model, self.get_name(), criterion, optimizer, writer, hyperparameters)

    def get_optimizer(self) -> torch.optim.Optimizer:
        if self._optim is None:
            self._optim = torch.optim.SGD(self.model.parameters(), lr=self.hyperparameters['LEARNING_RATE'],
                                          momentum=0.9)
        return self._optim

    def get_criterion(self):
        if self._criterion is None:
            self._criterion = nn.BCELoss()
        return self._criterion

    def train_loader(self, train_loader: DataLoader, test_loader: DataLoader) -> tuple:

        train_losses = np.zeros(self.hyperparameters['NUM_EPOCHS'])
        test_losses = np.zeros(self.hyperparameters['NUM_EPOCHS'])

        for epoch in tqdm(range(self.hyperparameters['NUM_EPOCHS']), desc='Train the model'):
            self.model.train()
            current_loss = 0.0
            tot_correct = 0

            optimizer = self.get_optimizer()
            criterion = self.get_criterion()

            for i, data in enumerate(train_loader, 0):
                inputs, targets = data
                inputs, targets = inputs.float().to(self.device), targets.float().to(self.device)
                targets = targets.reshape((targets.shape[0], 1))
                # Zero the gradients
                optimizer.zero_grad()
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                # Backpropagation
                loss.backward()
                optimizer.step()
                current_loss += loss.item()
                # Training Accuracy
                for out, tar in zip(outputs, targets):
                    if out > THRESHOLD:
                        out = 1
                    else:
                        out = 0
                    tot_correct += (out == tar).float().item()
                # Write the network graph at epoch 0, batch 0
                if epoch == 0 and i == 0:
                    self.writer.add_graph(self.model, input_to_model=data[0], verbose=False)

            train_losses[epoch] = current_loss
            accuracy = 100 * tot_correct / len(train_loader.dataset)
            self.writer.add_scalar("Stroke ANN - Loss/train", current_loss, epoch)
            self.writer.add_scalar("Stroke ANN - Accuracy/train", accuracy, epoch)

            # Evaluate accuracy at end of each epoch
            self.model.eval()
            val_loss = 0.0
            val_steps = 0
            for i, data in enumerate(test_loader, 0):
                inputs, targets = data
                inputs, targets = inputs.float().to(self.device), targets.float().to(self.device)
                targets = targets.reshape((targets.shape[0], 1))
                y_pred = self.model(inputs)
                test_loss = criterion(y_pred, targets)
                val_loss += test_loss.item()
                val_steps += 1

                acc = (y_pred.round() == targets).float().mean()
                acc = float(acc)
                if acc > best_acc:
                    best_acc = acc

            test_losses[epoch] = val_loss
            self.writer.add_scalar("Stroke ANN - Loss/test", val_loss, epoch)
            self.writer.add_scalar("Stroke ANN - Accuracy/test", acc, epoch)

        # Save the model at the end of the training (for future inference)
        self._save_model()
        self.writer.flush()
        self.writer.close()
        return train_losses, test_losses, best_acc

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor = None,
              y_test: torch.Tensor = None) -> Tuple[np.array, np.array]:
        pass

    def predict(self, X: torch.Tensor) -> np.array:
        self.model.eval()
        test_predictions = []
        with torch.no_grad():
            for i in range(len(X)):
                input_ = X[i].float()
                y_pred = self.model(input_)
                test_predictions.append(y_pred.item())
        return np.array(test_predictions)

    def get_name(self) -> str:
        return 'stroke_ann_' if self._name is None else self._name
