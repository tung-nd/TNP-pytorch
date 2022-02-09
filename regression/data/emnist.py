import argparse

import torch
import torchvision.datasets as tvds

from utils.paths import datasets_path
from utils.misc import gen_load_func

class EMNIST(tvds.EMNIST):
    def __init__(self, train=True, class_range=[0, 47], device='cpu', download=True):
        super().__init__(datasets_path, train=train, split='balanced', download=download)

        self.data = self.data.unsqueeze(1).float().div(255).transpose(-1, -2).to(device)
        self.targets = self.targets.to(device)

        idxs = []
        for c in range(class_range[0], class_range[1]):
            idxs.append(torch.where(self.targets==c)[0])
        idxs = torch.cat(idxs)

        self.data = self.data[idxs]
        self.targets = self.targets[idxs]

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
