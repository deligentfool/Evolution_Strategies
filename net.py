import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class es(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(es, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(self.observation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim)
        )

    def forward(self, observation):
        return self.net(observation)

    def count_params(self):
        count = 0
        for param in self.parameters():
            count += param.detach().view(-1).size(0)
        return count

    def get_params(self):
        return [(key, value) for key, value in zip(self.state_dict().keys(), self.state_dict().values())]