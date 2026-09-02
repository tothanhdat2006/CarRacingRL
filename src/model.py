import torch.nn as nn

class RuleModel(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        return -1, 1, 0