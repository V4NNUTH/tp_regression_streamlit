import torch
import torch.nn as nn

class SimpleLinearRegression(nn.Module):
    def __init__(self):
        super(SimpleLinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # 1 feature, 1 target

    def forward(self, x):
        return self.linear(x)