from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from .nn_model import SimpleModel
from .imitation_data import ImitationDatset

def eval_model(configs):
    model = SimpleModel()
    # https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
    model.load_state_dict(torch.load(configs.model_path, weights_only=True))
    model.eval()