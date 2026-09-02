import torch.nn as nn

class RuleModel(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        return -1, 1, 0


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # inp 96 x 96 x 3
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, padding = 1) # 96 x 96 x 16
        self.pool1 = nn.MaxPool2d(kernel_size = 2, padding = 0) # 48 x 48 x 16
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1) # 48 x 48 x 32
        self.pool2 = nn.MaxPool2d(kernel_size = 2, padding = 0) # 24 x 24 x 32

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(24 * 24 * 32, 128, bias=True)
        self.fc2 = nn.Linear(128, 3, bias=False)
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.reshape(-1,)

        x = self.relu(self.fc1(x))
        out = self.fc2(x)
        return out

if __name__ == "__main__":
    model = SimpleModel()
    import torch
    inp = torch.randn(3, 96, 96)
    out = model(inp)