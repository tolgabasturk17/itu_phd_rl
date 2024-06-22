import torch as T
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from model.ActorCriticAgent import ActorCriticAgent
from model.ActorCriticNetwork import ActorCriticNetwork
from environment.AirTrafficEnvironment import AirTrafficEnvironment
import grpc
import threading
from air_traffic_pb2 import EmptyRequest, AirTrafficComplexity
from air_traffic_pb2_grpc import AirTrafficServiceStub

class TCRManager:
    def __init__(self, config_data, grpc_channel, n_episodes=1000, learning_rate=0.001):
        # Initialize the scaler
        self.scaler = MinMaxScaler()

        self.max_sectors = 8

        # Create gRPC stub
        self.stub = AirTrafficServiceStub(grpc_channel)

        # Get initial metrics data from Java server
        initial_metrics_data = self._get_initial_metrics_data()

        # Initialize environment and agent
        self.env = AirTrafficEnvironment(config_data, initial_metrics_data, self.scaler, grpc_channel)
        self.agent = ActorCriticAgent(lr=learning_rate, input_dims=self.env.observation_space.shape[0],
                                      n_actions=self.env.action_space.n)
        self.n_episodes = n_episodes

        # Start streaming in a separate thread
        self.streaming_thread = threading.Thread(target=self._start_streaming)
        self.streaming_thread.start()

    def _get_initial_metrics_data(self):
        request = EmptyRequest()
        response = next(self.stub.StreamAirTrafficInfo(request))
        metrics_data = {
            'configuration_id': response.configuration_id,
            'cruising_sector_density': list(response.cruising_sector_density),
            'climbing_sector_density': list(response.climbing_sector_density),
            'descending_sector_density': list(response.descending_sector_density),
            'loss_of_separation': list(response.loss_of_separation),
            'speed_deviation': list(response.speed_deviation),
            'sector_entry': list(response.sector_entry),
            'airflow_complexity': list(response.airflow_complexity),
        }
        self._pad_metrics(metrics_data)
        self._update_scaler(metrics_data)
        return metrics_data

    def _start_streaming(self):
        request = EmptyRequest()
        responses = self.stub.StreamAirTrafficInfo(request)
        for response in responses:
            metrics_data = {
                'configuration_id': response.configuration_id,
                'cruising_sector_density': list(response.cruising_sector_density),
                'climbing_sector_density': list(response.climbing_sector_density),
                'descending_sector_density': list(response.descending_sector_density),
                'loss_of_separation': list(response.loss_of_separation),
                'speed_deviation': list(response.speed_deviation),
                'sector_entry': list(response.sector_entry),
                'airflow_complexity': list(response.airflow_complexity),
            }
            self._pad_metrics(metrics_data)
            self.env.metrics_data = metrics_data
            self._update_scaler(metrics_data)

    def _pad_metrics(self, metrics):
        for key in metrics.keys():
            if key == 'configuration_id':
                continue
            while len(metrics[key]) < self.max_sectors:
                metrics[key].append(0.0)

    def _update_scaler(self, metrics_data):
        flattened_metrics = []
        max_length = max(len(metric) for metric in metrics_data.values() if isinstance(metric, list))

        for step in range(max_length):
            step_metrics = []
            for metric_key, metric in metrics_data.items():
                if metric_key == 'configuration_id':
                    continue
                if isinstance(metric, list):
                    if step < len(metric):
                        step_metrics.append(metric[step])
                    else:
                        step_metrics.append(0)
            flattened_metrics.append([0] + step_metrics)  # Add a placeholder for current_configuration
        self.scaler.fit(flattened_metrics)

    def train_agent(self):
        for episode in range(self.n_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.agent.choose_action(state)
                state_, reward, done, info = self.env.step(action)
                self.agent.learn(state, reward, state_, done)
                state = state_

            # Optionally log progress
            if episode % 100 == 0:
                print(f"Episode {episode}/{self.n_episodes}")

        print("Training finished.")
        self._cleanup()

    def _cleanup(self):
        self.streaming_thread.join()

# Main code
if __name__ == "__main__":
    # Define config data (fill with actual data)
    config_data = {
        'Configurations': ['LTAAWCTA.CNF1', 'LTAAWCTA.CNF2', 'LTAAWCTA.CNF3A', 'LTAAWCTA.CNF3B', 'LTAAWCTA.CNF3C', 'LTAAWCTA.CNF3D',
                           'LTAAWCTA.CNF4A', 'LTAAWCTA.CNF4B', 'LTAAWCTA.CNF5A', 'LTAAWCTA.CNF5B', 'LTAAWCTA.CNF6A', 'LTAAWCTA.CNF7A', 'LTAAWCTA.CNF8A'],
        'current_configuration': 0  # Initial configuration index
    }

    # Create gRPC channel
    channel = grpc.insecure_channel('localhost:50051')

    # Create Main instance and start streaming
    main_instance = TCRManager(config_data, grpc_channel=channel)
    main_instance.train_agent()