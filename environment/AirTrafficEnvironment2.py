import gym
from gym import spaces
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import grpc
from air_traffic_pb2 import AirTrafficRequest
from air_traffic_pb2_grpc import AirTrafficServiceStub

class AirTrafficEnvironment2(gym.Env):
    def __init__(self, config_data, metrics_data, grpc_channel):
        super(AirTrafficEnvironment2, self).__init__()
        self.configurations = config_data['Configurations']
        self.metrics_data = metrics_data
        self.current_configuration = config_data['current_configuration']
        self.current_step = 0

        self.channel = grpc_channel
        self.stub = AirTrafficServiceStub(self.channel)

        self.max_sectors = 8
        self.num_features = 7  # Total number of metric categories (excluding configuration_id)

        self.observation_space = spaces.Box(low=0, high=1, shape=(56,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.configurations))

        self.state_scalers = self._initialize_state_scalers(metrics_data)
        self.cost_scalers = self._initialize_cost_scalers(metrics_data)

    def _initialize_state_scalers(self, metrics_data):

        # Minimum ve maksimum değerleri belirle
        min_max_values = {
            'cruising_sector_density': {
                'min': [0.0] * self.max_sectors,
                'max': [50.0] * self.max_sectors
            },
            'climbing_sector_density': {
                'min': [0.0] * self.max_sectors,
                'max': [50.0] * self.max_sectors
            },
            'descending_sector_density': {
                'min': [0.0] * self.max_sectors,
                'max': [50.0] * self.max_sectors
            },
            'loss_of_separation': {
                'min': [0.0] * self.max_sectors,
                'max': [10.0] * self.max_sectors
            },
            'speed_deviation': {
                'min': [0.0] * self.max_sectors,
                'max': [200.0] * self.max_sectors
            },
            'airflow_complexity': {
                'min': [-20.0] * self.max_sectors,
                'max': [20.0] * self.max_sectors
            },
            'sector_entry': {
                'min': [0.0] * self.max_sectors,
                'max': [30.0] * self.max_sectors
            }
        }

        state_scalers = {}
        for key in metrics_data:
            if key != 'configuration_id':
                scaler = MinMaxScaler()
                # Her parametre için minimum ve maksimum değerleri kullanarak fit et
                fit_data = np.array([min_max_values[key]['min'], min_max_values[key]['max']])
                scaler.fit(fit_data)
                state_scalers[key] = scaler
        return state_scalers

    def _initialize_cost_scalers(self, metrics_data):
        cost_scalers = {}
        total_metrics = self._calculate_total_metrics(metrics_data)
        for key in total_metrics:
            scaler = MinMaxScaler()
            data = np.array(total_metrics[key]).reshape(-1, 1)
            scaler.fit(data)
            cost_scalers[key] = scaler
        return cost_scalers

    def _flatten_metrics(self, metrics):
        flattened_metrics = []
        for metric_key, values in metrics.items():
            if metric_key == 'configuration_id':
                continue
            if len(values) < self.max_sectors:
                mean_value = sum(values) / len(values) if values else 0.0
                values.extend([mean_value] * (self.max_sectors - len(values)))
            flattened_metrics.extend(values)
        return flattened_metrics

    def _calculate_total_metrics(self, metrics):
        total_metrics = {
            'cruising_sector_density': sum(metrics['cruising_sector_density']),
            'climbing_sector_density': sum(metrics['climbing_sector_density']),
            'descending_sector_density': sum(metrics['descending_sector_density']),
            'loss_of_separation': sum(metrics['loss_of_separation']),
            'speed_deviation': sum(metrics['speed_deviation']),
            'airflow_complexity': sum(metrics['airflow_complexity']),
            'sector_entry': sum(metrics['sector_entry'])
        }
        return total_metrics

    def _scale_metrics(self, metrics):
        scaled_metrics = []
        for key, scaler in self.state_scalers.items():
            data = np.array(metrics[key]).reshape(1, -1)
            # Flatten ve pad işlemi
            scaled_data = scaler.transform(data).flatten()
            scaled_metrics.extend(scaled_data)
        return scaled_metrics

    def _scale_total_metrics(self, total_metrics):
        scaled_total_metrics = []
        for key, scaler in self.cost_scalers.items():
            data = np.array([total_metrics[key]]).reshape(-1, 1)
            scaled_data = scaler.transform(data).flatten()
            scaled_total_metrics.extend(scaled_data)
        return scaled_total_metrics

    def _get_state_size(self):
        return 1 + 7 * self.max_sectors

    def reset(self):
        self.current_step = 0
        self.current_configuration = 0
        self.state = self._get_state()
        return self.state

    def _get_state(self):
        metrics = self.metrics_data
        scaled_metrics = self._scale_metrics(metrics)
        return np.array(scaled_metrics)

    def _get_new_state(self, new_metrics):
        scaled_metrics = self._scale_metrics(new_metrics)
        return np.array(scaled_metrics)

    def step(self, action):
        self.current_configuration = action
        self.current_step += 1

        current_cost = self._calculate_cost(self.metrics_data)

        new_metrics = self._get_new_metrics_from_java(action)
        new_cost = self._calculate_cost(new_metrics)

        reward = new_cost - current_cost
        done = self.current_step >= len(self.metrics_data['cruising_sector_density'])
        self.state = self._get_new_state(new_metrics)
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
        scaled_metrics = self._scale_total_metrics(total_metrics)
        cost = sum(scaled_metrics)
        return cost

    #def _pad_metrics(self, metrics):
    #    for key in metrics.keys():
    #        if key == 'configuration_id':
    #            continue
    #        mean_value = sum(metrics[key]) / len(metrics[key]) if metrics[key] else 0.0
    #        while len(metrics[key]) < self.max_sectors:
    #            metrics[key].append(mean_value)

    def _pad_metrics(self, metrics):
        for key in metrics.keys():
            if key == 'configuration_id':
                continue
            while len(metrics[key]) < self.max_sectors:
                metrics[key].append(0.0)

    def render(self, mode='human', close=False):
        pass
