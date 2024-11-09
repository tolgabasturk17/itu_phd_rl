import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256):
        super(ActorCriticNetwork, self).__init__()

        # Shared Layers
        self.fc1 = nn.Linear(input_dims, fc1_dims)  # First shared layer
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)  # Second shared layer

        # Actor Layer
        self.pi = nn.Linear(fc2_dims, n_actions)  # Actor output layer

        # Critic Layer
        self.v = nn.Linear(fc2_dims, 1)  # Critic output layer

        # Optimizer and Device Setup
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):

        # Shared Forward Pass
        x = F.relu(self.fc1(state))  # Shared layer activation
        x = F.relu(self.fc2(x))  # Shared layer activation

        # Actor Forward Pass
        pi = self.pi(x)  # Actor output

        # Critic Forward Pass
        v = self.v(x)  # Critic output

        return pi, v  # Return both actor and critic outputs
