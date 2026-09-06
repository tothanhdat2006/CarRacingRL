import numpy as np
import torch

def select_greedy_action(obs, policy_model):
    """
    Args:
        obs (np.array) shape [96, 96, 3]: current environment
        policy_model (nn.Module): policy Deep Q-Network
    Return:
        action_id (int): the action index in actions array
    """
    pred_action = policy_model(obs)
    action_id = torch.argmax(pred_action, dim=-1).item()
    return action_id

def select_exploration_action(obs, policy_model, action_size, scheduler_exploration, t):
    """
    Args:
        obs (np.array) shape [96, 96, 3]: current environment
        policy_model (nn.Module): policy Deep Q-Network
        action_size (int): size of actions array
        scheduler_exploration (?): scheduler of action exploration based on timestep
        t (int): current timestep
    Return:
        action_id (int): the action index in actions array
    """
    pred_action = policy_model(obs)
    eps = scheduler_exploration.get_eps(t)
    random_number = np.random.randn(1)
    if random_number < eps:
        action_id = np.random.randint(0, action_size)
    else:
        action_id = torch.argmax(pred_action, dim=-1).item()
    return action_id


def get_actions(obs, policy_model, action_size, actions, scheduler_exploration, t, is_greedy=False):
    if is_greedy:
        action_id = select_greedy_action(obs, policy_model)
    else:
        action_id = select_exploration_action(obs, policy_model, action_size, scheduler_exploration, t)
    return actions[action_id], action_id

class Action_GetSet():
    def __init__(self):
        self.actions = [[-1.0, 0.05, 0], [1.0, 0.05, 0], [0, 0.5, 0], [0, 0, 1.0]]

    def set_actions(self, new_actions):
        self.actions = new_actions

    def get_actions(self):
        return self.actions