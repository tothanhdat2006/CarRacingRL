import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .nn_model import SimpleModel
from .imitation_data import ImitationDatset

def train_model(configs):
    print("Config data")
    imitation_dataset = ImitationDatset(configs.data.dataset_path)
    dataloader = DataLoader(imitation_dataset, shuffle=False)

    print("Config model")
    model = SimpleModel()
    optimizer = Adam(model.parameters(), lr=configs.train.lr)
    # scheduler = 
    criterion = nn.MSELoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    print("Training...")
    with tqdm(total=len(imitation_dataset) * configs.train.num_epochs, unit='its') as pbar:
        for epoch in range(configs.train.num_epochs):
            avg_loss = 0.0
            for batch in dataloader:
                obs_inp, acts_gt = batch['observations'].to(device), batch['actions'].to(device)
                pred = model(obs_inp)
                loss = criterion(pred, acts_gt) 

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(len(obs_inp))

    os.makedirs(configs.model.save_path, exist_ok=True)
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        },
        os.path.join(configs.model.save_path, configs.model.save_name)
    )
        