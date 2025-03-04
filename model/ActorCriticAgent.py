import torch as T
import torch.nn.functional as F
import numpy as np

from model.ActorCriticNetwork import ActorCriticNetwork

import numpy as np
import torch as T
import torch.nn.functional as F


class ActorCriticAgent():
    """
    Actor-Critic Agent using Deep Reinforcement Learning.

    This agent follows the Actor-Critic architecture where:
    - The Actor selects actions based on the policy network.
    - The Critic evaluates the state-action value to guide the Actor.

    The agent interacts with an environment, learns from rewards, and updates its policy
    using both policy gradient (Actor) and value estimation (Critic).

    Attributes:
    -----------
    gamma : float
        Discount factor for future rewards.
    actor_critic : ActorCriticNetwork
        Neural network model implementing both Actor and Critic.
    log_prob : torch.Tensor
        Stores the log probability of the last selected action.

    Author: Tolga BASTURK
    """
    def __init__(self, lr, input_dims, n_actions, gamma=0.99):
        """
        Initialize the Actor-Critic Agent.

        Parameters:
        - lr: Learning rate for the optimizer.
        - input_dims: Number of input features (state dimensions).
        - n_actions: Number of possible actions.
        - gamma: Discount factor for future rewards.
        """
        self.gamma = gamma  # Discount factor for future rewards
        self.actor_critic = ActorCriticNetwork(lr, input_dims, n_actions)  # Actor-Critic network initialization
        self.log_prob = None  # Store log probability of selected action for training

    def choose_action(self, observation):
        """
        Select an action based on the given observation.

        Parameters:
        - observation: The current state of the environment.

        Returns:
        - action: The chosen action (integer).
        """
        # Convert the observation to a NumPy array for compatibility
        observation_np = np.array(observation)

        # Convert the NumPy array into a PyTorch tensor and send it to the correct device (CPU/GPU)
        state = T.tensor(observation_np, dtype=T.float).unsqueeze(0).to(self.actor_critic.device)

        # Forward pass through the Actor-Critic network:
        # - probabilities: Actor's output (action probabilities)
        # - _: Critic's output (state value), unused in this function
        probabilities, _ = self.actor_critic.forward(state)

        # Apply softmax to convert raw scores into probabilities
        probabilities = F.softmax(probabilities, dim=1)

        # Define a categorical distribution based on the action probabilities
        action_probs = T.distributions.Categorical(probabilities)

        # Sample an action from the probability distribution
        action = action_probs.sample()

        # Store the log probability of the chosen action for use in learning
        self.log_prob = action_probs.log_prob(action)

        return action.item()  # Return the selected action as an integer

    def learn(self, state, reward, state_, done):
        """
        Perform the learning step to update the policy and value function.

        Parameters:
        - state: The current state before taking the action.
        - reward: The reward received after taking the action.
        - state_: The next state after taking the action.
        - done: Boolean flag indicating if the episode has ended.
        """
        # Reset gradients before optimization step
        self.actor_critic.optimizer.zero_grad()

        # Convert state and next state into NumPy arrays
        state_np = np.array(state)
        state__np = np.array(state_)

        # Convert them into PyTorch tensors and send them to the device (CPU/GPU)
        state = T.tensor(state_np, dtype=T.float).unsqueeze(0).to(self.actor_critic.device)
        state_ = T.tensor(state__np, dtype=T.float).unsqueeze(0).to(self.actor_critic.device)

        # Convert reward into a PyTorch tensor and send it to the device
        reward = T.tensor(reward, dtype=T.float).to(self.actor_critic.device)

        # Forward pass through the Actor-Critic network:
        # - _: Actor's output (not needed in this step)
        # - critic_value: Estimated value of the current state (V(s))
        _, critic_value = self.actor_critic.forward(state)

        # Forward pass for the next state (V(s'))
        _, critic_value_ = self.actor_critic.forward(state_)

        # Compute Temporal Difference (TD) Error:
        # If done=True, the next state’s value (V(s')) should be 0 (terminal state)
        delta = reward + self.gamma * critic_value_ * (1 - int(done)) - critic_value

        # Compute Actor loss:
        # The policy should encourage actions that lead to higher rewards (delta)
        actor_loss = -self.log_prob * delta

        # Compute Critic loss:
        # The critic should minimize the difference between estimated and actual values (Mean Squared Error)
        critic_loss = delta ** 2

        # Perform backpropagation to update both Actor and Critic networks
        (actor_loss + critic_loss).backward()

        # Apply gradient descent step to update the network parameters
        self.actor_critic.optimizer.step()

