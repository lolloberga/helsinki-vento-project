import numpy as np
import torch
from torch.utils.data import Dataset


class SmokeDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            self.X = torch.from_numpy(X.astype(np.float32))
            self.y = torch.from_numpy(y).type(torch.LongTensor)
        self.len = self.X.shape[0]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]