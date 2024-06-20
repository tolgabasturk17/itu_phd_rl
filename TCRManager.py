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
    def __init__(self, config_data,  metrics_data, grpc_channel, n_episodes=1000, learning_rate=0.001):
        # Initialize the scaler
        self.scaler = MinMaxScaler()

        self._update_scaler(metrics_data)

        # Initialize environment and agent
        self.env = AirTrafficEnvironment(config_data, metrics_data, self.scaler, grpc_channel)
        self.agent = ActorCriticAgent(lr=learning_rate, input_dims=self.env.observation_space.shape[0],
                                      n_actions=self.env.action_space.n)
        self.n_episodes = n_episodes

        # Start streaming in a separate thread
        self.streaming_thread = threading.Thread(target=self._start_streaming)
        self.streaming_thread.start()

    def _start_streaming(self):
        request = EmptyRequest()
        responses = self.stub.StreamAirTrafficInfo(request)
        for response in responses:
            metrics_data = {
                'cruising_sector_density': response.cruising_sector_density,
                'climbing_sector_density': response.climbing_sector_density,
                'descending_sector_density': response.descending_sector_density,
                'loss_of_separation': response.loss_of_separation,
                'speed_deviation': response.speed_deviation,
                'sector_entry': response.sector_entry,
                'airflow_complexity': response.airflow_complexity,
            }
            self.env.metrics_data = metrics_data
            self._update_scaler(metrics_data)

    def _update_scaler(self, metrics_data):
        flattened_metrics = []
        for step in range(len(metrics_data['cruising_sector_density'])):
            step_metrics = []
            for metric in metrics_data.values():
                step_metrics.extend(metric[step])
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


# Main code
if __name__ == "__main__":
    # Define config data (fill with actual data)
    config_data = {
        'Configurations': ['CNF1', 'CNF2', 'CNF3A', 'CNF4B'],
        'current_configuration': 0  # Initial configuration index
    }

    # Create gRPC channel
    channel = grpc.insecure_channel('localhost:50051')

    # Initial request to get metrics_data from Java server
    stub = AirTrafficServiceStub(channel)

    request = EmptyRequest()
    response = stub.StreamAirTrafficInfo(request)

    # Fill the metrics_data with the initial response
    metrics_data = {
        'cruising_sector_density': [response.cruising_sector_density],
        'climbing_sector_density': [response.climbing_sector_density],
        'descending_sector_density': [response.descending_sector_density],
        'loss_of_separation': [response.loss_of_separation],
        'speed_deviation': [response.speed_deviation],
        'airflow_complexity': [response.airflow_complexity],
        'sector_entry': [response.sector_entry]
    }

    # Create Main instance and start streaming
    main_instance = TCRManager(config_data, metrics_data, grpc_channel=channel)
    main_instance.train_agent()
