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
import logging

# Logger oluşturma
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TCRManager:
    def __init__(self, config_data, grpc_channel, n_episodes=3000, learning_rate=0.0005):
        self.max_sectors = 8
        self.state_scaler = MinMaxScaler()
        self.cost_scaler = MinMaxScaler()
        self.stub = AirTrafficServiceStub(grpc_channel)

        # Önce initial metrics data'yı alalım
        initial_metrics_data = self._get_initial_metrics_data()

        # Environment'ı oluşturalım
        self.env = AirTrafficEnvironment(config_data, initial_metrics_data, self.state_scaler, self.cost_scaler, grpc_channel)

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
                }
                self._pad_metrics(metrics_data)
                self.env.metrics_data = metrics_data
        except grpc.RpcError as e:
            logger.error(f"gRPC error during streaming: {e}")

    def _pad_metrics(self, metrics):
        for key in metrics.keys():
            if key == 'configuration_id':
                continue
            while len(metrics[key]) < self.max_sectors:
                metrics[key].append(0.0)

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
                logger.info(f"Episode {episode}/{self.n_episodes}")

        logger.info("Training finished.")
        self._cleanup()

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
        main_instance.train_agent()
    except Exception as e:
        logger.error(f"Exception occurred: {e}")
    finally:
        main_instance._cleanup()
