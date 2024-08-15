import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
class EEGImageDataset(Dataset):
    def __init__(self, eeg_path, images_path, labels_path, transform_eeg=None, transform_img=None):
        self.eegs = np.load(eeg_path)
        self.images = np.load(images_path)
        self.labels = np.load(labels_path)
        self.transform_eeg = transform_eeg
        self.transform_img = transform_img

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg = self.eegs[idx]
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform_eeg:
            eeg = self.transform_eeg(eeg)
        if self.transform_img:
            image = self.transform_img(image)

        return eeg, image, label