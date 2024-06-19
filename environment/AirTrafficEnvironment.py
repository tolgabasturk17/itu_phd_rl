import gym
from gym import spaces
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import grpc
from air_traffic_pb2 import AirTrafficRequest
from air_traffic_pb2_grpc import AirTrafficServiceStub

class AirTrafficEnvironment(gym.Env):
    def __init__(self, config_data, metrics_data, scaler, grpc_channel):
        super(AirTrafficEnvironment, self).__init__()
        self.configurations = config_data['Configurations']
        self.metrics_data = metrics_data
        self.scaler = scaler
        self.current_configuration = config_data['current_configuration']
        self.current_step = 0

        self.channel = grpc_channel
        self.stub = AirTrafficServiceStub(self.channel)

        max_sectors = max([len(self.metrics_data[metric][0]) for metric in self.metrics_data])
        self.observation_space = spaces.Box(low=0, high=1, shape=(1 + 5 * max_sectors,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.configurations))

    def _get_state_size(self):
        max_sectors = max([len(self.metrics_data[metric][0]) for metric in self.metrics_data])
        return 1 + 5 * max_sectors

    def reset(self):
        self.current_step = 0
        self.current_configuration = 0
        self.state = self._get_state()
        return self.state

    def _get_state(self):
        metrics = self.metrics_data
        step_metrics = []

        for metric in metrics.values():
            step_metrics.extend(metric[self.current_step])

        # Ensure the correct number of features
        required_features = self._get_state_size() - 1  # Subtract 1 for the current configuration
        while len(step_metrics) < required_features:
            step_metrics.append(0)  # Add zeros if there are not enough features
        step_metrics = step_metrics[:required_features]  # Trim if there are too many features

        # Include current configuration at the beginning
        full_features = [self.current_configuration] + step_metrics

        scaled_metrics = self.scaler.transform([full_features])[0]
        return np.array(scaled_metrics)

    def step(self, action):
        self.current_configuration = action
        self.current_step += 1

        new_metrics = self._get_new_metrics_from_java(action)
        reward = self._calculate_reward(new_metrics)
        done = self.current_step >= len(self.metrics_data['cruising_sector_density'])
        self.state = self._get_state()
        return self.state, reward, done, {}

    def _get_new_metrics_from_java(self, action):
        request = AirTrafficRequest(configuration_id=self.configurations[action])
        response = self.stub.GetAirTrafficInfo(request)
        new_metrics = {
            'cruising_sector_density': response.cruising_sector_density,
            'climbing_sector_density': response.climbing_sector_density,
            'descending_sector_density': response.descending_sector_density,
            'loss_of_separation': response.loss_of_separation,
            'speed_deviation': response.speed_deviation,
            'sector_entry': response.sector_entry,
            'airflow_complexity': response.airflow_complexity,
        }
        return new_metrics

    def _calculate_reward(self, new_metrics):
        total_cruising_density = sum(new_metrics['cruising_sector_density'])
        total_climbing_density = sum(new_metrics['climbing_sector_density'])
        total_descending_density = sum(new_metrics['descending_sector_density'])
        total_los = sum(new_metrics['loss_of_separation'])
        total_speed_deviation = sum(new_metrics['speed_deviation'])
        total_airflow_complexity = sum(new_metrics['airflow_complexity'])
        total_sector_entry = sum(new_metrics['sector_entry'])

        scaled_metrics = self.scaler.transform([[total_cruising_density, total_climbing_density, total_descending_density, total_los, total_speed_deviation, total_airflow_complexity, total_sector_entry]])[0]

        reward = -(scaled_metrics[0] + scaled_metrics[1] + scaled_metrics[2] + scaled_metrics[3] + scaled_metrics[4] + scaled_metrics[5] + scaled_metrics[6])
        return reward

    def render(self, mode='human', close=False):
        pass