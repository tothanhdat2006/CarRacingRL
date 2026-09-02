import numpy as np

import gymnasium as gym

from model import RuleModel

if __name__ == "__main__":
    render_mode = 'human'
    env = gym.make('CarRacing-v3', render_mode=render_mode)
    # print(env.observation_space.shape)
    # print(env.action_space.shape)
    model = RuleModel()

    for ep in range(5):
        obs, _ = env.reset()
        reward_per_ep = 0
        for t in range(100):
            steer, gas, brake = model(obs)
            action_np = np.array([steer, gas, brake])
            obs, reward, done, trunc, info = env.step(action_np)
            reward_per_ep += reward
        print('episode %d \t reward %f' % (ep, reward_per_ep))
    env.close()