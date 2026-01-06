import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.input_layer = nn.Linear(state_dim, 128)
        self.output_layer = nn.Linear(128, action_dim)

    
    def forward(self, state):
        x = F.relu(self.input_layer(state))
        probabilities = F.softmax(self.output_layer(x), dim=-1)
        return probabilities
    

class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.input_layer = nn.Linear(state_dim, 128)
        self.output_layer = nn.Linear(128, 1)

    
    def forward(self, state): 
        x = F.relu(self.input_layer(state))
        return self.output_layer(x)
    
