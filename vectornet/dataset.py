import numpy as np
import torch
from torch.utils.data import Dataset
import os
from IDX import IDX
from torch_geometric.data import Data
import random

class VNDataSet(Dataset):
    
    def __init__(self, files_dir):
        self.files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(files_dir) for f in filenames if os.path.splitext(f)[1] == '.npz']#[:600]
        random.shuffle(self.files)
        self.files = self.files[:240*4]
        data = np.load(self.files[0], allow_pickle=True)


    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):

        try:
            data = np.load(self.files[idx], allow_pickle=True)
            X = data["X"]
            ego_idx = data["X"][np.argmax(data["X"][:, IDX["ego"]] == 1), -1]
            return Data(
                x = torch.Tensor(X[:, :len(IDX) - 1]),
                i = torch.LongTensor(X[:, -1]),
                ego_idx = ego_idx,
                target_availabilities = torch.Tensor(data["target_availabilities"][None,]),
                target_positions = torch.Tensor(data["target_positions"][None,]),
                world_to_image = torch.Tensor(data["world_to_image"][None,]),
                centroid = torch.Tensor(data["centroid"][None,]),
                raster_from_agent = torch.Tensor(data["raster_from_agent"][None,]),
                name = self.files[idx]
            )
        except:
            data = np.load(self.files[0], allow_pickle=True)
            X = data["X"]
            ego_idx = data["X"][np.argmax(data["X"][:, IDX["ego"]] == 1), -1]
            return Data(
                x = torch.Tensor(X[:, :len(IDX) - 1]),
                i = torch.LongTensor(X[:, -1]),
                ego_idx = ego_idx,
                target_availabilities = torch.Tensor(data["target_availabilities"][None,]),
                target_positions = torch.Tensor(data["target_positions"][None,]),
                world_to_image = torch.Tensor(data["world_to_image"][None,]),
                centroid = torch.Tensor(data["centroid"][None,]),
                raster_from_agent = torch.Tensor(data["raster_from_agent"][None,]),
                name = self.files[idx]
            )