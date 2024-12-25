import torch as T
from model.ActorCriticAgent import ActorCriticAgent
from environment.AirTrafficEnvironment import AirTrafficEnvironment
import grpc
import threading
from air_traffic_pb2 import EmptyRequest, AirTrafficComplexity
from air_traffic_pb2_grpc import AirTrafficServiceStub

import logging

# Logger oluşturma
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TCRManager:
    """
    TCRManager is a class responsible for managing air traffic complexity calculations
    and interactions with a remote gRPC service that provides air traffic data.

    Attributes:
    -----------
    max_sectors : int
        The maximum number of sectors considered for air traffic analysis.

    stub : AirTrafficServiceStub
        A gRPC stub for interacting with the AirTrafficService to fetch air traffic data.

    env : AirTrafficEnvironment2
        An environment instance that simulates air traffic for the agent to interact with.

    agent : ActorCriticAgent
        The reinforcement learning agent that learns to manage air traffic complexity.

    n_episodes : int
        The number of training episodes for the reinforcement learning agent.

    running : bool
        A flag indicating whether the streaming process is currently running.

    streaming_thread : threading.Thread
        A thread responsible for streaming air traffic data from the gRPC server.

     Methods:
    --------
    __init__(config_data: dict, grpc_channel: grpc.Channel, n_episodes: int = 3000, learning_rate: float = 0.0005):
           Initializes the TCRManager with configuration data, a gRPC channel, number of episodes, and learning rate.

     _get_initial_metrics_data() -> dict:
         Fetches the initial metrics data from the gRPC service to initialize the environment.

    _start_streaming():
        Starts a background thread to continuously stream air traffic data from the gRPC service.

    _pad_metrics(metrics: dict):
        Pads the metrics data to ensure all lists have the same length as max_sectors.

    train_agent():
        Trains the reinforcement learning agent over a specified number of episodes.

    _cleanup():
        Cleans up resources and stops any running threads after training is complete.

    """
    def __init__(self, config_data, grpc_channel, n_episodes=100, learning_rate=0.0005):
        """
        Initializes the TCRManager with configuration data, a gRPC channel,
        number of training episodes, and learning rate.

        Parameters:
        -----------
        config_data : dict
            A dictionary containing configuration settings for the air traffic environment.

        grpc_channel : grpc.Channel
            A gRPC channel used to communicate with the remote AirTrafficService.

        n_episodes : int, optional
            The number of episodes for training the agent (default is 3000).

        learning_rate : float, optional
            The learning rate for the reinforcement learning agent (default is 0.0005).
        """
        self.max_sectors = 8
        self.stub = AirTrafficServiceStub(grpc_channel)

        # Fetch initial metrics data
        initial_metrics_data = self._get_initial_metrics_data()

        # Initialize the environment
        self.env = AirTrafficEnvironment(config_data, initial_metrics_data, grpc_channel)
        self.config_data = config_data

        # Initialize the agent
        self.agent = ActorCriticAgent(lr=learning_rate, input_dims=self.env.observation_space.shape[0], n_actions=self.env.action_space.n)
        self.n_episodes = n_episodes

        self.running = True
        self.streaming_thread = threading.Thread(target=self._start_streaming)
        self.streaming_thread.start()

    def _get_initial_metrics_data(self):
        """
        Fetches the initial metrics data from the gRPC service.

        Returns:
        --------
        dict
            A dictionary containing initial air traffic metrics data.
        """
        request = EmptyRequest()
        try:
            response = next(self.stub.StreamAirTrafficInfo(request))
        except grpc.RpcError as e:
            logger.error(f"Failed to get initial metrics data: {e}")
            raise e

        metrics_data = {
            'configuration_id': response.configuration_id,
            'time_interval': response.time_interval,
            'cruising_sector_density': list(response.cruising_sector_density),
            'climbing_sector_density': list(response.climbing_sector_density),
            'descending_sector_density': list(response.descending_sector_density),
            'loss_of_separation': list(response.loss_of_separation),
            'speed_deviation': list(response.speed_deviation),
            'sector_entry': list(response.sector_entry),
            'airflow_complexity': list(response.airflow_complexity),
            'number_of_controllers': response.number_of_controllers
        }
        self._pad_metrics(metrics_data)
        return metrics_data

    def _start_streaming(self):
        """
        Starts a thread to stream air traffic data continuously from the gRPC service.
        """
        request = EmptyRequest()
        try:
            responses = self.stub.StreamAirTrafficInfo(request)
            for response in responses:
                if not self.running:
                    break

                metrics_data = {
                    'configuration_id': response.configuration_id,
                    'time_interval': response.time_interval,
                    'cruising_sector_density': list(response.cruising_sector_density),
                    'climbing_sector_density': list(response.climbing_sector_density),
                    'descending_sector_density': list(response.descending_sector_density),
                    'loss_of_separation': list(response.loss_of_separation),
                    'speed_deviation': list(response.speed_deviation),
                    'sector_entry': list(response.sector_entry),
                    'airflow_complexity': list(response.airflow_complexity),
                    'number_of_controllers': response.number_of_controllers
                }
                self._pad_metrics(metrics_data)
                self.env.metrics_queue.put(metrics_data)

        except grpc.RpcError as e:
            logger.error(f"gRPC error during streaming: {e}")

    def _pad_metrics(self, metrics):
        """
        Ensures that all metric lists have the same length by padding them.

        Parameters:
            ----------
        metrics : dict
            A dictionary containing lists of metrics to be padded.
        """
        for key in metrics.keys():
            if key == 'configuration_id' or key == 'time_interval' or key == 'number_of_controllers':
                continue
            while len(metrics[key]) < self.max_sectors:
                metrics[key].append(0.0)

    def train_agent(self):
        for episode in range(self.n_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            step_count = 0

            while not done:
                action = self.agent.choose_action(state)
                state_, reward, done, info = self.env.step(action)

                # Öğrenme için agent aksiyonunun sonucunda elde edilen state_
                learn_state = state_

                # Döngünün devamı için gerçek dünya verisinden elde edilen state_
                true_state = self.env._get_state()

                self.agent.learn(state, reward, learn_state, done)
                state = true_state

                total_reward += reward
                step_count += 1

                logger.info(f"Episode: {episode}, Step: {step_count}, Agent choice: {self.config_data['Configurations'][action]}, Reward: {reward}, Total Reward: {total_reward}")

            logger.info(f"Episode {episode} finished. Total reward: {total_reward}, Total steps: {step_count}")
            logger.info(f"Saving model at episode {episode}")
            self.save_model(f'actor_critic_model_{episode}.pth')

        logger.info("Training finished.")
        self.save_model('actor_critic_model_final.pth')
        self._cleanup()

    def save_model(self, filepath='actor_critic_model.pth'):
        T.save(self.agent.actor_critic.state_dict(), filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath='actor_critic_model.pth'):
        self.agent.actor_critic.load_state_dict(T.load(filepath))
        self.agent.actor_critic.eval()  # Modeli değerlendirme moduna geçir
        logger.info(f"Model loaded from {filepath}")

    def test_agent(self, n_episodes=100):
        for episode in range(n_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            step_count = 0

            while not done:
                action = self.agent.choose_action(state)
                state_, reward, done, info = self.env.step(action)
                total_reward += reward
                logger.info(f"Episode: {episode}, Step: {step_count}, Agent choice: {self.config_data['Configurations'][action]}, Reward: {reward}, Total Reward: {total_reward}")
                state = self.env._get_state()
                step_count += 1

            logger.info(f"Test Episode {episode} finished. Total reward: {total_reward}, Total steps: {step_count}")

    def _cleanup(self):
        self.running = False
        if self.streaming_thread.is_alive():
            self.streaming_thread.join()

if __name__ == "__main__":
    config_data = {
        'Configurations': ['LTAAWCTA.CNF1', 'LTAAWCTA.CNF2', 'LTAAWCTA.CNF3A', 'LTAAWCTA.CNF3B', 'LTAAWCTA.CNF3C', 'LTAAWCTA.CNF3D', 'LTAAWCTA.CNF4A', 'LTAAWCTA.CNF4B', 'LTAAWCTA.CNF5A', 'LTAAWCTA.CNF5B', 'LTAAWCTA.CNF6A', 'LTAAWCTA.CNF7A', 'LTAAWCTA.CNF8A'],
        'current_configuration': 0
    }

    channel = grpc.insecure_channel('localhost:50051')

    main_instance = TCRManager(config_data, grpc_channel=channel)
    try:
        #main_instance.load_model('actor_critic_model_0.pth')  # Modeli yüklemek için kullanılabilir
        main_instance.train_agent()
        #main_instance.load_model('actor_critic_model_final.pth')  # Modeli yüklemek için kullanılabilir
        #main_instance.test_agent()
    except Exception as e:
        logger.error(f"Exception occurred: {e}")
    finally:
        main_instance._cleanup()
