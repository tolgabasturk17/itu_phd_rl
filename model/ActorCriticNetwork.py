import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ActorCriticNetwork(nn.Module):
    """
        This class defines an Actor-Critic network architecture used in reinforcement learning.
        It consists of shared layers for feature extraction, and separate layers for actor and critic outputs.

        Author: Tolga BASTURK
    """
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256):
        """
            Initializes the Actor-Critic network with the specified parameters.

            Args:
                lr (float): Learning rate for the optimizer.
                input_dims (int): Number of input features for the network.
                n_actions (int): Number of possible actions the agent can take.
                fc1_dims (int, optional): Number of neurons in the first fully connected layer (default: 256).
                fc2_dims (int, optional): Number of neurons in the second fully connected layer (default: 256).
        """
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
        """
            Performs a forward pass through the network to generate both the policy (actor output)
            and the value (critic output) for the given input state.

            Args:
                state (torch.Tensor): The input state represented as a tensor.

            Returns:
                tuple: A tuple containing:
                    - pi (torch.Tensor): The policy output (action probabilities).
                    - v (torch.Tensor): The value output (state value).
        """

        # Shared Forward Pass
        # Pass the input state through the first shared layer and apply ReLU activation.
        x = F.relu(self.fc1(state))  # Shared layer activation
        # Pass the output of the first shared layer through the second shared layer.
        x = F.relu(self.fc2(x))  # Shared layer activation

        # Actor Forward Pass
        # Generate the action probabilities (policy) from the actor layer.
        pi = self.pi(x)  # Actor output

        # Critic Forward Pass
        # Generate the value of the state from the critic layer.
        v = self.v(x)  # Critic output

        # Return both the policy (actor output) and the value (critic output).
        return pi, v  # Return both actor and critic outputs
