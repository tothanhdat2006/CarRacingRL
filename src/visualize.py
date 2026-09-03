import os
import numpy as np

import torch
import torchvision.transforms.v2 as T
import gymnasium as gym

from .nn_model import SimpleModel

transforms = T.Compose([
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def visualize_model(configs):
    model = SimpleModel()
    # https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
    checkpoint = torch.load(os.path.join(configs.model.save_path, configs.model.save_name), weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    render_mode = 'human'
    env = gym.make('CarRacing-v3', render_mode=render_mode)
    obs, _ = env.reset()
    total_reward = 0
    num_steps = 1000
    for t in range(num_steps):
        obs = torch.as_tensor(obs).permute(2, 0, 1)
        obs = transforms(obs)
        steer, gas, brake = model(obs)
        steer, gas, brake = steer.item(), gas.item(), brake.item()
        # print(steer, gas, brake)
        obs, reward, done, trunc, info = env.step(np.array([steer, gas, brake]))
        total_reward += reward

    print(total_reward / num_steps)