import grpc
from sklearn.preprocessing import MinMaxScaler
from model.ActorCriticAgent import ActorCriticAgent
from model.ActorCriticNetwork import ActorCriticNetwork
from environment.AirTrafficEnvironment import AirTrafficEnvironment

class TCRManager:
    def __init__(self, config_data, metrics_data, n_episodes=1000, learning_rate=0.001, grpc_channel=None):
        self.scaler = MinMaxScaler()
        flattened_metrics = []
        for step in range(len(metrics_data['sector_density'])):
            step_metrics = []
            for metric in metrics_data.values():
                step_metrics.extend(metric[step])
            flattened_metrics.append([0] + step_metrics)

        self.scaler.fit(flattened_metrics)
        self.env = AirTrafficEnvironment(config_data, metrics_data, self.scaler, grpc_channel)
        self.agent = ActorCriticAgent(lr=learning_rate, input_dims=len(self.env.observation_space.low),
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
            if episode % 100 == 0:
                print(f"Episode {episode}/{self.n_episodes}")

        print("Training finished.")

if __name__ == "__main__":
    metrics_data = {
        'sector_density': [
            [1, 2, 3], [4, 5, 6],
        ],
        'loss_of_separation': [
            [0, 1, 0], [1, 0, 1],
        ],
        'speed_deviation': [
            [10, 20, 30], [15, 25, 35],
        ],
        'airflow_complexity': [
            [0.1, 0.2, 0.3], [0.4, 0.5, 0.6],
        ],
        'sector_entry': [
            [2, 3, 4], [3, 4, 5],
        ]
    }

    config_data = {
        'Configurations': ['CNF1', 'CNF2', 'CNF3A', 'CNF4B'],
        'current_configuration': 0
    }

    channel = grpc.insecure_channel('localhost:50051')
    main_instance = TCRManager(config_data, metrics_data, grpc_channel=channel)
    main_instance.train_agent()