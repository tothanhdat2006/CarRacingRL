from datetime import time
import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam

from .replay_buffer import ReplayBuffer
from .nn_model import DQN, q_step, update_target_net
from .action import Action_GetSet, get_actions
from .schedule import LinearSchedule
from .utils import visualize_train, get_state

def train_model(env, configs, new_actions = []):
    print("Config data")
    replay_buffer = ReplayBuffer(configs.data.buffer_size)

    print("Config model")
    action_getset = Action_GetSet()
    if len(new_actions) > 0:
        action_getset.set_actions(new_actions) # override new actions for new games
    actions = action_getset.get_actions()
    action_size = len(actions)
    target_model = DQN(action_size)
    policy_model = DQN(action_size)
    optimizer = Adam(policy_model.parameters(), lr=configs.train.lr)
    scheduler_exploration = LinearSchedule(
        schedule_timesteps=int(configs.train.exploration_fraction * configs.train.total_timesteps),
        initial_p=1.0,
        final_p=configs.train.exploration_final_eps
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_model.to(device)
    policy_model.to(device)
    policy_model.train()
    target_model.load_state_dict(policy_model.state_dict())
    target_model.eval()

    print("Training...")
    losses = []
    eps_reward=  [0.0]
    start_time = time.time()
    with tqdm(total=configs.train.total_timesteps, unit='its') as pbar:
        for t in range(configs.train.total_timesteps):
            # https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
            # 1. Take actions according to eps-greedy policy
            chosen_action, action_id = get_actions(obs, policy_model, action_size, actions, scheduler_exploration, t, is_greedy=False)
            
            # 2. Sample new state and store to replay buffer
            ## 2.1 frame-skipping
            """
            Atari paper state that:
                Following previous approaches to playing Atari 2600 games,we also use a simple
                frame-skipping technique15. More precisely, the agent sees and selects actions on
                every kth frame instead of every frame, and its last action is repeated on skipped
                frames
            """
            for frame in range(configs.train.num_action_repeat):
                new_obs, reward, term, trunc, _ = env.step(chosen_action)
                done = term or trunc
                eps_reward[-1] += reward
                if done:
                    break
            ## 2.2 store in replay buffer D
            """
            Atari paper state that:
                learning directly from consecutive samples is inefficient since there are correlations between samples
                and learning on policy will produce data follow policy -> data are less diverse and may stuck in local optima
            """
            new_obs_state = get_state(new_obs)
            replay_buffer.add(obs_state, action_id, reward, new_obs_state)
            obs = new_obs

            ## 2.3 check terminal or truncated
            if done:
                obs, _ = env.reset()
                obs_state = get_state(obs)
                eps_reward.append(0.0)

            # 3 + 4 + 5
            if t > configs.train.learning_starts and t % configs.train.train_freq == 0:
                loss = q_step(
                    policy_model, 
                    target_model, 
                    optimizer, 
                    replay_buffer, 
                    configs.train.batch_size, 
                    configs.train.gamma, 
                    device, 
                    configs.train.use_DoubleQ)
                losses.append(loss)

            # 6. update target network
            if t > configs.train.learning_starts and t % configs.train.target_update_freq == 0:
                update_target_net(policy_model, target_model)

    end_time = time.time()
    print(f"Train for total {end_time - start_time:.5f} sec")
    os.makedirs(configs.model.save_path, exist_ok=True)
    torch.save(
        {
            'model_state_dict': policy_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        },
        os.path.join(configs.model.save_path, configs.model.save_name)
    )
    visualize_train(eps_reward, losses, configs)

        