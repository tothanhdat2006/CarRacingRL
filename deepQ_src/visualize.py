
import os
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.v2 as T
import gymnasium as gym

from .action import select_greedy_action, Action_GetSet
from .nn_model import DQN

transforms = T.Compose([
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def visualize_model(configs, new_actions = []):
    action_getset = Action_GetSet()
    if len(new_actions) > 0:
        action_getset.set_actions(new_actions) # override new actions for new strategy
    actions = action_getset.get_actions()

    model = DQN(len(actions))
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
        obs = torch.as_tensor(obs[np.newaxis]).permute(0, 3, 1, 2)
        obs = transforms(obs)
        action = actions[select_greedy_action(obs, model)]
        # print(steer, gas, brake)
        obs, reward, done, trunc, info = env.step(np.array(action))
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