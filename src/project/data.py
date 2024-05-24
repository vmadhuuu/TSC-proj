import pandas as pd
import numpy as np
from scipy.io.arff import loadarff
import torch
from torch.utils.data import Dataset

class Data(Dataset):
    """
    Implements a PyTorch Dataset class
    """
    def __init__(self, dataset, testing):
        train_data = pd.DataFrame(
            loadarff(f'data/{dataset}_TRAIN.arff')[0])
        dtypes = {i: np.float32 for i in train_data.columns[:-1]}
        dtypes.update({train_data.columns[-1]: int})
        train_data = train_data.astype(dtypes)

        self.x = train_data.iloc[:, :-1].values
        self.y = train_data.iloc[:, -1].values

        self.mean = np.mean(self.x, axis=0)
        self.std = np.std(self.x, axis=0)

        if testing:
            test_data = pd.DataFrame(
                loadarff(f'data/{dataset}_TEST.arff')[0])
            dtypes = {i: np.float32 for i in test_data.columns[:-1]}
            dtypes.update({test_data.columns[-1]: int})
            test_data = test_data.astype(dtypes)

            self.x = test_data.iloc[:, :-1].values
            self.y = test_data.iloc[:, -1].values

        class_labels = np.unique(self.y)
        if (class_labels != np.arange(len(class_labels))).any():
            new_labels = {old_label: i for i, old_label in enumerate(class_labels)}
            self.y = np.array([new_labels[old_label] for old_label in self.y])

        self.x = (self.x - self.mean) / self.std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y
