import os
import numpy as np
from PIL import Image

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

    all_obs = []

    render_mode = 'rgb_array'
    env = gym.make('CarRacing-v3', render_mode=render_mode)
    obs, _ = env.reset()
    all_obs.append(obs)

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
        all_obs.append(obs)

    # https://stackoverflow.com/questions/63047707/how-do-i-convert-a-numpy-array-to-a-gif
    all_frames = [Image.fromarray(obs) for obs in all_obs]
    all_frames[0].save(os.path.join(configs.vis.save_path, "visualization.gif"), 
                    save_all=True, 
                    append_images=all_frames[1:],
                    duration=33.33,
                    loop=0)

    print(total_reward / num_steps)