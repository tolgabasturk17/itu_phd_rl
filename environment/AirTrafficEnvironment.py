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

        self.max_sectors = 8
        self.num_features = 7  # Total number of metric categories (excluding configuration_id)

        self.observation_space = spaces.Box(low=0, high=1, shape=(56,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.configurations))

        self.scaler = self._initialize_scaler(metrics_data)

    def _initialize_scaler(self, metrics_data):
        # Tüm özellikler için 0 ile doldurulmuş veriler oluştur
        flattened_metrics = []
        for metric_key, values in metrics_data.items():
            if metric_key == 'configuration_id':
                continue
            # Her metrik için 8 değere kadar doldurma yap
            if len(values) < self.max_sectors:
                values = values + [0.0] * (self.max_sectors - len(values))
            flattened_metrics.extend(values)

        # İlk eğitim için veri seti oluştur
        training_data = [flattened_metrics for _ in range(10)]  # 10 örnek kullanarak scaler'ı eğit

        scaler = MinMaxScaler()
        scaler.fit(training_data)
        return scaler

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
        full_features = []

        # Configuration ID'yi dışarıda tut
        config_id = metrics.pop('configuration_id', 'default-config')

        # Her bir metrik için işlem yap
        for metric_key, values in metrics.items():
            # Eğer değerlerin uzunluğu max_sectors'dan az ise, eksik kısımları sıfır ile doldur
            if len(values) < self.max_sectors:
                values.extend([0.0] * (self.max_sectors - len(values)))
            # Değerleri full_features listesine ekle
            full_features.extend(values)

        # Ensure full_features is of the correct length
        while len(full_features) < self.num_features * self.max_sectors:
            full_features.append(0.0)

        full_features = full_features[:self.num_features * self.max_sectors]

        # Verileri ölçeklendir
        scaled_features = self.scaler.transform([full_features])[0]
        return np.array(scaled_features)

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
            'configuration_id': response.configuration_id,
            'cruising_sector_density': list(response.cruising_sector_density),
            'climbing_sector_density': list(response.climbing_sector_density),
            'descending_sector_density': list(response.descending_sector_density),
            'loss_of_separation': list(response.loss_of_separation),
            'speed_deviation': list(response.speed_deviation),
            'sector_entry': list(response.sector_entry),
            'airflow_complexity': list(response.airflow_complexity),
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
