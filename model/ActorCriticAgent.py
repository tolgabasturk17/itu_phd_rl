import torch as T
import torch.nn.functional as F

from model.ActorCriticNetwork import ActorCriticNetwork

class ActorCriticAgent():
    def __init__(self, lr, input_dims, n_actions, gamma=0.99):
        self.gamma = gamma
        self.actor_critic = ActorCriticNetwork(lr, input_dims, n_actions)
        self.log_prob = None

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor_critic.device)
        probabilities, _ = self.actor_critic.forward(state)
        probabilities = F.softmax(probabilities, dim=1)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        self.log_prob = action_probs.log_prob(action)
        return action.item()

    def learn(self, state, reward, state_, done):
        self.actor_critic.optimizer.zero_grad()
        state = T.tensor([state], dtype=T.float).to(self.actor_critic.device)
        state_ = T.tensor([state_], dtype=T.float).to(self.actor_critic.device)
        reward = T.tensor(reward, dtype=T.float).to(self.actor_critic.device)
        _, critic_value = self.actor_critic.forward(state)
        _, critic_value_ = self.actor_critic.forward(state_)
        delta = reward + self.gamma * critic_value_ * (1 - int(done)) - critic_value
        actor_loss = -self.log_prob * delta
        critic_loss = delta**2
        (actor_loss + critic_loss).backward()
        self.actor_critic.optimizer.step()