import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
class EEGDataset(Dataset):
    def __init__(self, eeg_path, labels_path, transform_eeg=None):
        self.eegs = np.load(eeg_path)
        self.labels = np.load(labels_path)
        self.transform_eeg = transform_eeg

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg = self.eegs[idx]
        label = self.labels[idx]

        if self.transform_eeg:
            eeg = self.transform_eeg(eeg)
        return eeg, label