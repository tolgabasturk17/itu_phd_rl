import gym
from gym import spaces
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from air_traffic_pb2 import AirTrafficRequest
from air_traffic_pb2_grpc import AirTrafficServiceStub

class AirTrafficEnvironment2(gym.Env):
    """
    AirTrafficEnvironment2 is a custom OpenAI Gym environment designed for air traffic management simulations.
    It interacts with a remote gRPC service to fetch air traffic data and simulates various configurations of
    air traffic scenarios for reinforcement learning agents.

    Parameters
    ----------
    config_data : dict
        Configuration data specifying the available configurations and the current configuration index.
    metrics_data : dict
        Initial metrics data used to set up the environment state.
    grpc_channel : grpc.Channel
        A gRPC channel for communication with the remote AirTrafficService.

    Attributes
    ----------
    configurations : list
        List of available configurations for air traffic management.
    metrics_data : dict
        Current metrics data representing the state of the environment.
    current_configuration : int
        Index of the currently selected configuration.
    current_step : int
        Counter for the current step within an episode.
    channel : grpc.Channel
        gRPC channel used for communication with the AirTrafficService.
    stub : AirTrafficServiceStub
        Stub for making remote procedure calls to the AirTrafficService.
    max_sectors : int
        Maximum number of sectors in the air traffic simulation.
    num_features : int
        Number of features (metrics categories) excluding configuration ID.
    observation_space : gym.spaces.Box
        Observation space defining the range and shape of the environment's state representation.
    action_space : gym.spaces.Discrete
        Action space defining the range of valid actions the agent can take.
    state_scalers : dict
        Scalers for normalizing state metrics.
    cost_scalers : dict
        Scalers for normalizing cost metrics.

    Methods
    -------
    _initialize_state_scalers(metrics_data)
        Initializes scalers for state metrics based on min-max values.
    _initialize_cost_scalers(metrics_data)
        Initializes scalers for cost metrics based on min-max values.
    _calculate_controller_load(metrics)
        Calculates the load on controllers based on various metrics.
    _scale_metrics(metrics)
        Scales the input metrics using pre-defined scalers.
    _scale_total_metrics(total_metrics)
        Scales the total metrics using pre-defined scalers.
    _get_state_size()
        Returns the size of the state vector.
    reset()
        Resets the environment to its initial state and returns the initial observation.
    _get_state()
        Retrieves the current state of the environment, scaled to fit within the observation space.
    _get_new_state(new_metrics)
        Generates a new state representation based on updated metrics.
    step(action)
        Takes a step in the environment using the specified action, returning the next state, reward,
        done flag, and additional info.
    _get_new_metrics_from_java(action)
        Fetches new metrics data from the gRPC service based on the provided action.
    _calculate_cost(metrics)
        Calculates the cost based on the given metrics with specified weights.
    _pad_metrics(metrics)
        Pads metrics lists to ensure consistent length across all metric categories.
    render(mode='human', close=False)
        Renders the environment. Currently a placeholder function.
    """

    def __init__(self, config_data, metrics_data, grpc_channel):
        """
        Initializes the AirTrafficEnvironment2 with configuration data, initial metrics data,
        and a gRPC communication channel.

        Parameters
        ----------
        config_data : dict
            Configuration data specifying available configurations and the current configuration index.
        metrics_data : dict
            Initial metrics data used to initialize the environment state.
        grpc_channel : grpc.Channel
            gRPC channel for communication with the remote AirTrafficService.
        """
        super(AirTrafficEnvironment2, self).__init__()
        self.configurations = config_data['Configurations']
        self.metrics_data = metrics_data
        self.current_configuration = config_data['current_configuration']
        self.current_step = 0

        self.channel = grpc_channel
        self.stub = AirTrafficServiceStub(self.channel)

        self.max_sectors = 8
        self.num_features = 7  # Total number of metric categories (excluding configuration_id)

        self.observation_space = spaces.Box(low=0, high=1, shape=(57,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.configurations))

        self.state_scalers = self._initialize_state_scalers()
        self.cost_scalers = self._initialize_cost_scalers()

    def _initialize_state_scalers(self):
        """
        Initializes the MinMax scalers for each state metric based on predefined minimum and maximum values.

        Parameters
        ----------
        metrics_data : dict
            The initial metrics data used to determine the range for scaling.

        Returns
        -------
        dict
            A dictionary of MinMaxScaler objects for each metric.
        """
        # Define minimum and maximum values for each metric
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
                'max': [250.0] * self.max_sectors
            },
            'airflow_complexity': {
                'min': [-200.0] * self.max_sectors,
                'max': [20.0] * self.max_sectors
            },
            'sector_entry': {
                'min': [0.0] * self.max_sectors,
                'max': [50.0] * self.max_sectors
            },
            'number_of_controllers': {
                'min': [0.0],
                'max': [16.0]
            }
        }

        state_scalers = {}
        for key in min_max_values:
            if key != 'configuration_id':
                scaler = MinMaxScaler()
                fit_data = np.array([min_max_values[key]['min'], min_max_values[key]['max']])
                scaler.fit(fit_data)
                state_scalers[key] = scaler
        return state_scalers

    def _initialize_cost_scalers(self):
        """
        Initializes the MinMax scalers for cost metrics based on predefined minimum and maximum values.

        Parameters
        ----------
        metrics_data : dict
            The initial metrics data used to determine the range for scaling.

        Returns
        -------
        dict
            A dictionary of MinMaxScaler objects for each cost metric.
        """
        min_max_values = {
            'cruising_sector_density': {
                'min': [0.0],
                'max': [20.0]
            },
            'climbing_sector_density': {
                'min': [0.0],
                'max': [20.0]
            },
            'descending_sector_density': {
                'min': [0.0],
                'max': [20.0]
            },
            'loss_of_separation': {
                'min': [0.0],
                'max': [10.0]
            },
            'speed_deviation': {
                'min': [0.0],
                'max': [200.0]
            },
            'airflow_complexity': {
                'min': [0.0],
                'max': [250.0]
            },
            'sector_entry': {
                'min': [0.0],
                'max': [20.0]
            },
            'number_of_controllers': {
                'min': [0.0],
                'max': [16.0]
            }
        }
        cost_scalers = {}
        for key in min_max_values:
            scaler = MinMaxScaler()
            fit_data = np.array([min_max_values[key]['min'], min_max_values[key]['max']])
            scaler.fit(fit_data)
            cost_scalers[key] = scaler
        return cost_scalers

    def _calculate_controller_load(self, metrics):
        """
        Calculates the controller load based on various metrics, adjusting for negative and positive values.

        Parameters
        ----------
        metrics : dict
            Dictionary containing the current metrics for the environment.

        Returns
        -------
        dict
            A dictionary representing the calculated load for each metric.
        """
        controller_load = {}

        # Calculate load for each relevant metric
        for key in ['cruising_sector_density', 'climbing_sector_density', 'descending_sector_density',
                    'speed_deviation', 'sector_entry']:
            non_zero_values = [value for value in metrics[key] if value > 0]
            controller_load[key] = np.mean(non_zero_values) if non_zero_values else 0.0

        # Separate calculation for airflow complexity
        if 'airflow_complexity' in metrics:
            positive_values = [value for value in metrics['airflow_complexity'] if value > 5]
            negative_values = [value for value in metrics['airflow_complexity'] if value < -5]

            positive_mean = np.mean(positive_values) if positive_values else 0.0
            negative_mean = np.mean(negative_values) if negative_values else 0.0

            # Adjust negative impact
            controller_load['airflow_complexity'] = (1.5 * abs(negative_mean)) - positive_mean

        # Total loss of separation
        controller_load['loss_of_separation'] = np.sum(metrics['loss_of_separation'])
        controller_load['number_of_controllers'] = metrics['number_of_controllers']

        return controller_load

    def _scale_metrics(self, metrics):
        """
        Scales the input metrics using predefined scalers.

        Parameters
        ----------
        metrics : dict
            The raw metrics data to be scaled.

        Returns
        -------
        list
            A flattened list of scaled metric values.
        """
        scaled_metrics = []
        for key, scaler in self.state_scalers.items():
            data = np.array(metrics[key]).reshape(1, -1)
            scaled_data = scaler.transform(data).flatten()
            scaled_metrics.extend(scaled_data)
        return scaled_metrics

    def _scale_total_metrics(self, total_metrics):
        """
        Scales the total metrics using predefined scalers.

        Parameters
        ----------
        total_metrics : dict
            The total metrics to be scaled.

        Returns
        -------
        list
            A flattened list of scaled total metric values.
        """
        scaled_total_metrics = []
        for key, scaler in self.cost_scalers.items():
            data = np.array([total_metrics[key]]).reshape(-1, 1)
            scaled_data = scaler.transform(data).flatten()
            scaled_total_metrics.extend(scaled_data)
        return scaled_total_metrics

    def _get_state_size(self):
        """
        Returns the size of the state vector.

        Returns
        -------
        int
            The size of the state vector.
        """
        return 1 + 7 * self.max_sectors

    def reset(self):
        """
        Resets the environment to its initial state.

        Returns
        -------
        np.ndarray
            The initial observation of the environment.
        """
        self.current_step = 0
        self.current_configuration = 0
        self.state = self._get_state()
        return self.state

    def _get_state(self):
        """
        Retrieves the current state of the environment, scaled to fit within the observation space.

        Returns
        -------
        np.ndarray
            The current state observation.
        """
        metrics = self.metrics_data
        scaled_metrics = self._scale_metrics(metrics)
        return np.array(scaled_metrics)

    def _get_new_state(self, new_metrics):
        """
        Generates a new state representation based on updated metrics.

        Parameters
        ----------
        new_metrics : dict
            The updated metrics data.

        Returns
        -------
        np.ndarray
            The new state observation.
        """
        scaled_metrics = self._scale_metrics(new_metrics)
        return np.array(scaled_metrics)

    def step(self, action):
        """
        Takes a step in the environment using the specified action, returning the next state, reward,
        done flag, and additional info.

        Parameters
        ----------
        action : int
            The action chosen by the agent.

        Returns
        -------
        tuple
            A tuple containing the next state (np.ndarray), reward (float), done (bool), and additional info (dict).
        """
        self.current_configuration = action
        self.current_step += 1

        current_cost = self._calculate_cost(self.metrics_data)

        new_metrics = self._get_new_metrics_from_java(action)
        new_cost = self._calculate_cost(new_metrics)

        reward = current_cost - new_cost

        # Finish at the end of the day
        done = self.current_step % 135 == 0

        self.state = self._get_new_state(new_metrics)
        return self.state, reward, done, {}

    def _get_new_metrics_from_java(self, action):
        """
        Fetches new metrics data from the gRPC service based on the provided action.

        Parameters
        ----------
        action : int
            The action chosen by the agent.

        Returns
        -------
        dict
            The updated metrics data fetched from the gRPC service.
        """
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
            'number_of_controllers': response.number_of_controllers
        }
        self._pad_metrics(new_metrics)
        return new_metrics

    def _calculate_cost(self, metrics):
        """
        Calculates the cost based on the given metrics with specified weights.

        Parameters
        ----------
        metrics : dict
            The current metrics data.

        Returns
        -------
        float
            The calculated cost value.
        """
        total_metrics = self._calculate_controller_load(metrics)
        scaled_metrics = self._scale_total_metrics(total_metrics)

        # Weights for specific metrics
        weights = {
            'loss_of_separation': 2.0,
            'airflow_complexity': 2.0,
            'number_of_controllers': 1.0
        }

        cost = 0
        for key, value in zip(total_metrics.keys(), scaled_metrics):
            weight = weights.get(key, 1.0)
            cost += value * weight

        return cost

    def _pad_metrics(self, metrics):
        """
        Pads metrics lists to ensure consistent length across all metric categories.

        Parameters
        ----------
        metrics : dict
            The metrics data to be padded.
        """
        for key in metrics.keys():
            if key == 'configuration_id' or key == 'number_of_controllers':
                continue
            while len(metrics[key]) < self.max_sectors:
                metrics[key].append(0.0)

    def render(self, mode='human', close=False):
        """
        Renders the environment. Currently a placeholder function.

        Parameters
        ----------
        mode : str, optional
            The mode in which to render the environment. Default is 'human'.
        close : bool, optional
            Flag indicating whether to close the rendering. Default is False.
        """
        pass
