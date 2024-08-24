import torch as T
from model.ActorCriticAgent import ActorCriticAgent
from environment.AirTrafficEnvironment2 import AirTrafficEnvironment2
import grpc
import threading
from air_traffic_pb2 import EmptyRequest, AirTrafficComplexity
from air_traffic_pb2_grpc import AirTrafficServiceStub

import logging

# Logger oluşturma
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TCRManager:
    def __init__(self, config_data, grpc_channel, n_episodes=3000, learning_rate=0.0005):
        self.max_sectors = 8
        self.stub = AirTrafficServiceStub(grpc_channel)

        # Önce initial metrics data'yı alalım
        initial_metrics_data = self._get_initial_metrics_data()

        # Environment'ı oluşturalım
        self.env = AirTrafficEnvironment2(config_data, initial_metrics_data, grpc_channel)

        self.agent = ActorCriticAgent(lr=learning_rate, input_dims=self.env.observation_space.shape[0], n_actions=self.env.action_space.n)
        self.n_episodes = n_episodes

        self.running = True
        self.streaming_thread = threading.Thread(target=self._start_streaming)
        self.streaming_thread.start()

    def _get_initial_metrics_data(self):
        request = EmptyRequest()
        try:
            response = next(self.stub.StreamAirTrafficInfo(request))
        except grpc.RpcError as e:
            logger.error(f"Failed to get initial metrics data: {e}")
            raise e

        metrics_data = {
            'configuration_id': response.configuration_id,
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
        request = EmptyRequest()
        try:
            responses = self.stub.StreamAirTrafficInfo(request)
            for response in responses:
                if not self.running:
                    break
                metrics_data = {
                    'configuration_id': response.configuration_id,
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
                self.env.metrics_data = metrics_data
        except grpc.RpcError as e:
            logger.error(f"gRPC error during streaming: {e}")

    def _pad_metrics(self, metrics):
        for key in metrics.keys():
            if key == 'configuration_id' or key == 'number_of_controllers':
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

                logger.info(f"Episode: {episode}, Step: {step_count}, Action: {action}, Reward: {reward}")

            logger.info(f"Episode {episode} finished. Total reward: {total_reward}, Total steps: {step_count}")

            if episode % 1 == 0:
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
                state, reward, done, info = self.env.step(action)
                total_reward += reward
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
        #main_instance.load_model('actor_critic_model_16.pth')  # Modeli yüklemek için kullanılabilir
        main_instance.train_agent()
        # main_instance.load_model('actor_critic_model_final.pth')  # Modeli yüklemek için kullanılabilir
        # main_instance.test_agent()
    except Exception as e:
        logger.error(f"Exception occurred: {e}")
    finally:
        main_instance._cleanup()
