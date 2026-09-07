import numpy as np
class ReplayBuffer():
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
        self.current_idx = 0

    def __len__(self):
        return len(self.buffer)

    def add_sample(self, obs, act, reward, obs_next, done):
        """
        Add new samples to buffer in cyclic manner
        """
        if len(self.buffer) < self.max_size: # init first: append
            self.buffer.append((obs, act, reward, obs_next, done))
        else: # cycle back to beginning: update instead of append
            self.current_idx = self.current_idx % self.max_size
            self.buffer[self.current_idx] = (obs, act, reward, obs_next, done)
        self.current_idx += 1

    def _encode_samples(self, idx_list):
        """
        Encode the samples before return

        Args:
            idx_list (np.array): list of indices of samples to return
        Returns:
            (s, a, r, s', done) (tuple[np.array, np.array, np.array, np.array, np.array])
        """
        obs, act, reward, obs_next, done_list = [], [], [], [], []
        for idx in idx_list:
            s, a, r, s_next, done = self.buffer[idx]
            obs.append(s)
            act.append(np.array(a))
            reward.append(r)
            obs_next.append(s_next)
            done_list.append(done)
        return np.array(obs), np.array(act), np.array(reward), np.array(obs_next), np.array(done_list)
    
    def sample(self, num_samples):
        """
        Main sample logic that randomly draw (s, a, r, s') from buffer
        """
        idx_list = np.random.randint(0, len(self.buffer), num_samples)
        return self._encode_samples(idx_list)