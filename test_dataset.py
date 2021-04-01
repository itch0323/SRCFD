import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.002, 0.199, -5.710])
std = np.array([0.676, 1.901, 35.074])

class ImageDataset(Dataset):
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.Normalize(mean, std),
            ]
        )

        self.lr = torch.from_numpy(np.load("./test/CFD032_list.npy").astype(np.float32)).clone()
        self.hr = torch.from_numpy(np.load("./test/CFD128_list.npy").astype(np.float32)).clone()

    def __getitem__(self, index):

        self.img_lr = self.transform(self.lr[index])
        self.img_hr = self.transform(self.hr[index])

        return {"lr": self.img_lr, "hr": self.img_hr}

    def __len__(self):
        return len(self.lr)
