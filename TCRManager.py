import torch as T
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from model.ActorCriticAgent import ActorCriticAgent
from model.ActorCriticNetwork import ActorCriticNetwork
from environment.AirTrafficEnvironment import AirTrafficEnvironment
import grpc

class TCRManager:
    def __init__(self, config_data, metrics_data, grpc_channel, n_episodes=1000, learning_rate=0.001):
        # Initialize the scaler
        self.scaler = MinMaxScaler()

        # Flatten metrics data for scaling
        flattened_metrics = []
        for step in range(len(metrics_data['cruising_sector_density'])):
            step_metrics = []
            for metric in metrics_data.values():
                step_metrics.extend(metric[step])
            flattened_metrics.append([0] + step_metrics)  # Add a placeholder for current_configuration

        self.scaler.fit(flattened_metrics)

        # Initialize environment and agent
        self.env = AirTrafficEnvironment(config_data, metrics_data, self.scaler, grpc_channel)
        self.agent = ActorCriticAgent(lr=learning_rate, input_dims=self.env.observation_space.shape[0],
                                      n_actions=self.env.action_space.n)
        self.n_episodes = n_episodes

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
    # Example metrics data (fill with actual data)
    metrics_data = {
        'cruising_sector_density': [
            [1, 2, 3],  # Step 1
            [4, 5, 6],  # Step 2
            # Add the rest of your cruising_sector_density data here
        ],
        'climbing_sector_density': [
            [1, 2, 3],  # Step 1
            [4, 5, 6],  # Step 2
            # Add the rest of your climbing_sector_density data here
        ],
        'descending_sector_density': [
            [1, 2, 3],  # Step 1
            [4, 5, 6],  # Step 2
            # Add the rest of your descending_sector_density data here
        ],
        'loss_of_separation': [
            [0, 1, 0],  # Step 1
            [1, 0, 1],  # Step 2
            # Add the rest of your loss_of_separation data here
        ],
        'speed_deviation': [
            [10, 20, 30],  # Step 1
            [15, 25, 35],  # Step 2
            # Add the rest of your speed_deviation data here
        ],
        'airflow_complexity': [
            [0.1, 0.2, 0.3],  # Step 1
            [0.4, 0.5, 0.6],  # Step 2
            # Add the rest of your airflow_complexity data here
        ],
        'sector_entry': [
            [2, 3, 4],  # Step 1
            [3, 4, 5],  # Step 2
            # Add the rest of your sector_entry data here
        ]
    }

    # Define config data (fill with actual data)
    config_data = {
        'Configurations': ['CNF1', 'CNF2', 'CNF3A', 'CNF4B'],
        'current_configuration': 0  # Initial configuration index
    }

    # Create gRPC channel
    channel = grpc.insecure_channel('localhost:50051')

    # Create Main instance and train agent
    main_instance = TCRManager(config_data, metrics_data, grpc_channel=channel)
    main_instance.train_agent()
