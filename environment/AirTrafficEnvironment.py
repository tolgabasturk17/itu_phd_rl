import gym
from gym import spaces
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class AirTrafficEnvironment(gym.Env):
    def __init__(self, config_data, metrics_data, scaler, max_input_dim):
        self.config_data = config_data
        self.metrics_data = metrics_data
        self.scaler = scaler
        self.max_input_dim = max_input_dim

        # Action and observation space
        self.action_space = spaces.Discrete(len(config_data['Configurations']))
        self.observation_space = spaces.Box(low=0, high=1, shape=(1 + len(metrics_data) * max_input_dim,),
                                            dtype=np.float32)

        self.current_step = 0
        self.current_configuration = config_data['current_configuration']

    def reset(self):
        self.current_step = 0
        self.current_configuration = self.config_data['current_configuration']
        return self._get_observation()

    def step(self, action):
        self.current_configuration = action
        self.current_step += 1

        done = self.current_step >= len(self.metrics_data['sector_density'])

        reward = self._calculate_reward()
        info = {}

        return self._get_observation(), reward, done, info

    def _get_observation(self):
        step_metrics = []
        for metric_name, metric in self.metrics_data.items():
            padded_metric = metric[self.current_step] + [0] * (self.max_input_dim - len(metric[self.current_step]))
            step_metrics.extend(padded_metric)
        observation = [self.current_configuration] + step_metrics
        observation = self.scaler.transform([observation])[0]
        return np.array(observation, dtype=np.float32)

    def _calculate_reward(self, action):
        # Implement your reward calculation logic
        # Here, you should use the selected action (configuration)
        # and compare the actual metrics with expected ones to calculate the reward
        metrics = []
        for metric in self.metrics_data.values():
            metrics.extend(metric[self.current_step])

        sector_density = sum(self.metrics_data['sector_density'][self.current_step])
        total_los = sum(self.metrics_data['loss_of_separation'][self.current_step])
        total_speed_deviation = sum(self.metrics_data['speed_deviation'][self.current_step])
        total_airflow_complexity = sum(self.metrics_data['airflow_complexity'][self.current_step])
        total_sector_entry = sum(self.metrics_data['sector_entry'][self.current_step])

        # Define weighting factors for each metric
        alpha, beta, gamma, delta, epsilon = 1, 1, 1, 1, 1  # Adjust these values as needed

        reward = -(alpha * sector_density + beta * total_los + gamma * total_speed_deviation +
                   delta * total_airflow_complexity + epsilon * total_sector_entry)
        return reward

    def render(self, mode='human', close=False):
        pass
