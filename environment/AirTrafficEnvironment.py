import gym
from gym import spaces
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import grpc
from air_traffic_pb2 import AirTrafficRequest
from air_traffic_pb2_grpc import AirTrafficServiceStub

class AirTrafficEnvironment(gym.Env):
    def __init__(self, config_data, metrics_data, state_scaler, cost_scaler, grpc_channel):
        super(AirTrafficEnvironment, self).__init__()
        self.configurations = config_data['Configurations']
        self.metrics_data = metrics_data
        self.state_scaler = state_scaler
        self.cost_scaler = cost_scaler
        self.current_configuration = config_data['current_configuration']
        self.current_step = 0

        self.channel = grpc_channel
        self.stub = AirTrafficServiceStub(self.channel)

        self.max_sectors = 8
        self.num_features = 7  # Total number of metric categories (excluding configuration_id)

        self.observation_space = spaces.Box(low=0, high=1, shape=(56,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.configurations))

        self.state_scaler = self._initialize_state_scaler(metrics_data)
        self.cost_scaler = self._initialize_cost_scaler(metrics_data)

    def _initialize_state_scaler(self, metrics_data):
        flattened_metrics = self._flatten_metrics(metrics_data)
        training_data = [flattened_metrics for _ in range(10)]
        scaler = MinMaxScaler()
        scaler.fit(training_data)
        return scaler

    def _initialize_cost_scaler(self, metrics_data):
        total_metrics = self._calculate_total_metrics(metrics_data)
        training_data = [total_metrics for _ in range(10)]
        scaler = MinMaxScaler()
        scaler.fit(training_data)
        return scaler

    def _flatten_metrics(self, metrics):
        flattened_metrics = []
        for metric_key, values in metrics.items():
            if metric_key == 'configuration_id':
                continue
            if len(values) < self.max_sectors:
                values.extend([0.0] * (self.max_sectors - len(values)))
            flattened_metrics.extend(values)
        return flattened_metrics

    def _calculate_total_metrics(self, metrics):
        total_metrics = [
            sum(metrics['cruising_sector_density']),
            sum(metrics['climbing_sector_density']),
            sum(metrics['descending_sector_density']),
            sum(metrics['loss_of_separation']),
            sum(metrics['speed_deviation']),
            sum(metrics['airflow_complexity']),
            sum(metrics['sector_entry'])
        ]
        return total_metrics

    def _get_state_size(self):
        self.max_sectors = max([len(metric) if isinstance(metric, list) else 1 for metric in self.metrics_data.values() if metric != self.metrics_data.get('configuration_id')])
        return 1 + 6 * self.max_sectors

    def reset(self):
        self.current_step = 0
        self.current_configuration = 0
        self.state = self._get_state()
        return self.state

    def _get_state(self):
        metrics = self.metrics_data
        full_features = self._flatten_metrics(metrics)
        scaled_features = self.state_scaler.transform([full_features])[0]
        return np.array(scaled_features)

    def step(self, action):
        self.current_configuration = action
        self.current_step += 1

        current_cost = self._calculate_cost(self.metrics_data)

        new_metrics = self._get_new_metrics_from_java(action)
        new_cost = self._calculate_cost(new_metrics)

        reward = current_cost - new_cost

        done = self.current_step >= len(self.metrics_data['cruising_sector_density'])
        self.state = self._get_state()
        return self.state, reward, done, {}

    def _get_new_metrics_from_java(self, action):
        request = AirTrafficRequest(configuration_id=self.configurations[action])
        response = self.stub.GetAirTrafficInfo(request)
        new_metrics = {
            'configuration_id': response.configuration_id,
            'cruising_sector_density': list(response.cruising_sector_density),
            'climbing_sector_density': list(response.climbing_sector_density),
            'descending_sector_density': list(response.descending_sector_density),
            'loss_of_separation': list(response.loss_of_separation),
            'speed_deviation': list(response.speed_deviation),
            'sector_entry': list(response.sector_entry),
            'airflow_complexity': list(response.airflow_complexity),
        }
        self._pad_metrics(new_metrics)
        return new_metrics

    def _calculate_cost(self, metrics):
        total_metrics = self._calculate_total_metrics(metrics)
        scaled_metrics = self.cost_scaler.transform([total_metrics])[0]
        cost = sum(scaled_metrics)
        return cost

    def _pad_metrics(self, metrics):
        for key in metrics.keys():
            if key == 'configuration_id':
                continue
            while len(metrics[key]) < self.max_sectors:
                metrics[key].append(0.0)

    def render(self, mode='human', close=False):
        pass
