import os
from tqdm import tqdm

import torch
import gymnasium as gym

from .nn_model import SimpleModel

def visualize_model(configs):
    model = SimpleModel()
    # https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
    model.load_state_dict(torch.load(configs.model_path, weights_only=True))
    model.eval()

    render_mode = 'human'
    env = gym.make('CarRacing-v3', render_mode=render_mode)