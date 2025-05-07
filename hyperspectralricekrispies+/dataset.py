import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import os
import h5py
import numpy as np

class KrispyDataset(Dataset):
    def __init__(self, root_dir, transform=None, square_size=64, nclasses=90):
        """
        Args:
            root_dir (str): Directory with the HDF5 files (create two, one for val and one for train)
            transform (callable, optional): Optional transform to be applied on a sample
            square_size (int): The length of both sides to pad the image to (therefore images must be less than or equal)
        """
        self.nclasses = nclasses
        self.root_dir = root_dir
        self.transform = transform
        self.square_size = square_size
        self.files = [os.path.join(root_dir, f) for f in sorted(os.listdir(root_dir)) if f.endswith('.h5')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        with h5py.File(file_path, 'r') as file:
            image = torch.from_numpy(np.array(file["image"])).transpose(2,0) # Transpose to c,x,y instead of x,y,c
            class_id = file["image"].attrs["class_id"]
            short_name = file["image"].attrs["short_name"]

        # Use one-hot encoding to get the label
        labels = torch.zeros((self.nclasses), dtype=torch.float64)
        labels[class_id] = 1.0
        
        # Make sure the images aren't already larger than we want them to be
        assert(image.shape[1] <= self.square_size)
        assert(image.shape[2] <= self.square_size)
        # Pad them to the same size
        image = nn.functional.pad(image, ( self.square_size - image.shape[2], 0, self.square_size - image.shape[1], 0 ), value=0.0)

        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'labels': labels,
            'class_id': class_id,
            'short_name': short_name
        }
