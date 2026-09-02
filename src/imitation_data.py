import os
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T

class ImitationDatset(Dataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.files = os.listdir(dataset_path)
        print("Loading dataset actions and observations...")
        self.actions = [np.load(os.path.join(dataset_path, file)) for file in self.files if file.startswith("action_")]
        self.obs = [np.load(os.path.join(dataset_path, file)) for file in self.files if file.startswith("observation_")]

        # https://docs.pytorch.org/vision/stable/transforms.html
        self.transforms = T.Compose([
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        obs = torch.as_tensor(self.obs[idx]).permute(2, 0, 1) # transform require C H W instead of H W C
        acts = torch.as_tensor(self.actions[idx])

        obs = self.transforms(obs)

        return {
            "actions": acts,
            "observations": obs
        }

if __name__ == "__main__":
    imitation_dataset = ImitationDatset("./data")
    data = imitation_dataset[0]
    print(data["observations"])