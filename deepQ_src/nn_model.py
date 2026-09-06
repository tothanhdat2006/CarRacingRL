import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T

class DQN(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim
        # inp 96 x 96 x 3
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, padding = 1) # 96 x 96 x 16
        self.pool1 = nn.MaxPool2d(kernel_size = 2, padding = 0) # 48 x 48 x 16
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1) # 48 x 48 x 32
        self.pool2 = nn.MaxPool2d(kernel_size = 2, padding = 0) # 24 x 24 x 32

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(24 * 24 * 32, 128, bias=True)
        self.fc2 = nn.Linear(128, action_dim, bias=False)

    def forward(self, obs):
        """
        Original DQN playing Atari said: "There are several possible ways of parameterizing Q using a
        neural network. Because Q maps history–action pairs to scalar estimates of their
        Q-value, the history andthe action have been usedas inputsto the neuralnetwork
        by some previous approaches24,26. The main drawback of this type of architecture
        is that a separate forward pass is required to compute the Q-value of each action,
        resulting in a cost that scales linearly with the number of actions" 
        
        Since driving a car requires continuous and long history of driving, 
        stacking previous frames for input is not a good idea for device with low memory

        Args:
            observation (torch.tensor): environment states
        Returns:
            Q-values (torch.tensor): q-values for all actions
        """
        x = self.relu(self.conv1(obs))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        # x = x.reshape(x.shape[0], -1)
        x = x.flatten(1)

        x = self.relu(self.fc1(x))
        out = self.fc2(x)
        return out
        

# https://docs.pytorch.org/vision/stable/transforms.html
transforms = T.Compose([
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
def q_step(
    policy_model, 
    target_model, 
    optimizer, 
    replay_buffer, 
    batch_size, 
    gamma, 
    device, 
    use_DoubleQ
):
    """
    Perform one q-learning step
    https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

    Args:
        policy_model (nn.Module)
        target_model (nn.Module)
        optimizer (torch.optim)
        replay_buffer (object)
        batch_size (int) 
        gamma (float)
        device (torch.device)
        use_DoubleQ (bool)
    """

    # 3. Sample mini-batch from replay buffer
    batch = replay_buffer.sample(batch_size)
    s, a, r, s_next, done = batch
    s_tensor = torch.tensor(s, device=device).permute(0, 3, 1, 2)
    a_tensor = torch.tensor(a, dtype=torch.long, device=device)
    r_tensor = torch.tensor(r, dtype=torch.float32, device=device)
    s_next_tensor = torch.tensor(s_next, device=device).permute(0, 3, 1, 2)
    done_tensor = torch.tensor(done, dtype=torch.bool, device=device)

    s_transformed = transforms(s_tensor)
    s_next_transformed = transforms(s_next_tensor)
    print(s_transformed.shape)
    return 0.0

    # 4. Compute Q targets using old parameters
    with torch.no_grad():
        q_targets = target_model(s_next_transformed) # (B, A)
        if not use_DoubleQ:
            max_next_q_target = q_targets.max(dim=-1).values
            y_targets = r_tensor + (~done_tensor).float() * gamma * max_next_q_target
        else:
            # Select using current model
            policy_next_actions = policy_model(s_next_transformed) # (B, A)
            next_actions = policy_next_actions.max(dim=-1, keepdim=True).indices
            # Evaluate using target model
            future_term = q_targets.gather(dim=1, index=next_actions).squeeze(1) 
            y_targets = r_tensor + (~done_tensor).float() * gamma * future_term

    # and 5. Optimize between target Q and policy Q
    q_values = policy_model(s_transformed) # (B, A)
    q_action_values = q_values.gather(dim=1, index=a_tensor.unsqueeze(1)) # (B, )
    q_action_values = q_action_values.squeeze(1) # (B)

    loss = F.mse_loss(q_action_values, y_targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def update_target_net(policy_model, target_model):
    target_model.load_state_dict(policy_model.state_dict())