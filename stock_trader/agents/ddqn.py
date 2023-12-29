"""
Double Deep Q-Learning Network (DDQN) agent.
"""
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# pylint: disable=import-error
from agents.utils.sum_tree import SumTree

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DNN(nn.Module):
    """
    Simple Deep Neural Network

    Parameters
    ----------
    input_size : int
        The input size of the neural network.
    hidden_size : int
        The hidden size of the neural network.
    output_size : int
        The output size of the neural network.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Forward pass of the neural network.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(DNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Parameters
        ----------
        x : torch.Tensor
            The input to the neural network.

        Returns
        -------
        torch.Tensor
            The output of the neural network.
        """
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output_layer(x)
        return x


class PrioritedReplayBuffer:
    """
    Prioritized Replay Buffer

    Parameters
    ----------
    buffer_size : int
        The size of the replay buffer.
    alpha : float
        The alpha value for prioritized replay.
    buffer : list
        The replay buffer.
    priorities: np.array
        The priorities for the replay buffer.
    position : int
        The current position in the replay buffer.
    """

    def __init__(self, buffer_size: int, alpha: float = 0.6):
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.buffer = SumTree(buffer_size)
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.position = 0

    def add(self, error: float, sample: tuple):
        """
        Add a sample to the replay buffer.

        Parameters
        ----------
        error : float
            The error for the sample.
        sample : tuple
            The sample to add to the replay buffer.
        """
        priority = (abs(error) + 1e-5) ** self.alpha
        self.buffer.add(priority, sample)

    def sample(self, batch_size: int, beta: float) -> tuple:
        """
        Sample from the replay buffer.

        Parameters
        ----------
        batch_size : int
            The batch size for training.
        beta : float
            The beta value for prioritized replay.

        Returns
        -------
        tuple
            The samples from the replay buffer.
        np.array
            The indices of the samples.
        np.array
            The weights for the samples.
        """
        indices = np.zeros(batch_size, dtype=np.int32)
        samples = []
        weights = np.zeros((batch_size), dtype=np.float32)

        total_priority = self.buffer.total_priority()
        priority_segment = total_priority / batch_size

        for i in range(batch_size):
            a = priority_segment * i
            b = priority_segment * (i + 1)

            value = np.random.uniform(a, b)
            index, priority, sample = self.buffer.get_leaf(value)
            weights[i] = (priority / total_priority) ** beta
            indices[i] = index
            samples.append(sample)

        weights /= np.max(weights)
        return samples, indices, weights

    def update_priorities(self, indices: np.array, errors: np.array):
        """
        Update the priorities of the replay buffer.

        Parameters
        ----------
        indices : np.array
            The indices of the samples.
        errors : np.array
            The errors for the samples.
        """
        for i, error in zip(indices, errors):
            priority = (abs(error) + 1e-5) ** self.alpha
            self.buffer.update(i, priority)


class DDQN:
    """
    Deep Double Q-Learning Network
    """

    def __init__(self, state_size: int, action_size: int, config: dict):
        """
        Initialize the DDQN agent.

        Parameters
        ----------
        state_size : int
            The state size of the environment.
        action_size : int
            The action size of the environment.
        config : dict
            The configuration for the agent.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.config = config

        self.memory = PrioritedReplayBuffer(config["buffer_size"], config["alpha"])
        self.gamma = config["gamma"]  # discount rate
        self.epsilon = config["epsilon"]  # exploration rate
        self.epsilon_min = config["epsilon_min"]
        self.epsilon_decay = config["epsilon_decay"]
        self.learning_rate = config["learning_rate"]
        self.tau = config["tau"]
        self.beta = config["beta"]

        self.q_network = DNN(state_size, config["hidden_size"], action_size)
        self.target_network = DNN(state_size, config["hidden_size"], action_size)
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=self.learning_rate
        )

        # Initializing target network weights with the Q network weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        # Setting target network in evaluation mode
        self.target_network.eval()

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=self.gamma
        )

        # Initialize SummaryWriter
        self.writer = SummaryWriter(config["logs_path"])

        # For calculating average Q value
        self.previous_q_value = None
        self.q_value_diff = []

    def act(self, state: np.array) -> int:
        """
        Choose an action based on the current state and epsilon-greedy policy.

        Parameters
        ----------
        state : np.array
            The current state of the environment.
        """
        self.q_network.eval()
        if np.random.rand() > self.epsilon:  # exploitation
            self.q_network.eval()
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
            with torch.no_grad():
                action_values: DNN = self.q_network(state)
            self.q_network.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:  # exploration
            return np.random.choice(np.arange(self.action_size))

    def replay(self, batch_size: int = None) -> float:
        """
        Experience replay with prioritized replay buffer.
        """

        batch_size = self.config["batch_size"] if batch_size is None else batch_size

        if len(self.memory.buffer) < batch_size:
            return

        samples, indices, weights_tensor = self._sample_from_memory(batch_size)
        (
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
        ) = self._unpack_samples(samples)
        loss, q_targets = self._compute_loss(
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
            weights_tensor,
        )

        self.q_network.train()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1)
        self.optimizer.step()

        errors = (
            torch.abs(
                q_targets
                - self.q_network(states_tensor)
                .gather(1, actions_tensor.unsqueeze(1))
                .squeeze(1)
            )
            .data.cpu()
            .numpy()
        )
        self.update_priorities(indices, errors)

        self.update_epsilon()

        self.scheduler.step()

        return loss.item()

    def _sample_from_memory(
        self, batch_size: int
    ) -> tuple[
        list[tuple[np.ndarray, int, float, np.ndarray, bool]], np.ndarray, torch.Tensor
    ]:
        """
        Sample a minibatch from the replay buffer.

        Parameters
        ----------
        batch_size : int
            The batch size for training.

        Returns
        -------
        tuple[list[tuple[np.ndarray, int, float, np.ndarray, bool]], np.ndarray, torch.Tensor]
            The samples from the replay buffer.
        """
        samples, indices, weights = self.memory.sample(batch_size, self.beta)
        weights_tensor: torch.Tensor = torch.tensor(weights, dtype=torch.float).to(
            device
        )
        return samples, indices, weights_tensor

    def _unpack_samples(
        self, samples: list[tuple[np.ndarray, int, float, np.ndarray, bool]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Unpack the samples from the replay buffer.

        Parameters
        ----------
        samples : list[tuple[np.ndarray, int, float, np.ndarray, bool]]
            The samples from the replay buffer.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            The unpacked samples from the replay buffer.
        """
        states, actions, rewards, next_states, dones = zip(*samples)
        states_tensor: torch.Tensor = (
            torch.from_numpy(np.array(states)).float().to(device)
        )
        actions_tensor: torch.Tensor = (
            torch.from_numpy(np.array(actions)).long().to(device)
        )
        rewards_tensor: torch.Tensor = (
            torch.from_numpy(np.array(rewards)).float().to(device)
        )
        next_states_tensor: torch.Tensor = (
            torch.from_numpy(np.array(next_states)).float().to(device)
        )
        dones_tensor: torch.Tensor = torch.from_numpy(np.array(dones)).int().to(device)
        return (
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
        )

    def _compute_loss(
        self,
        states_tensor: torch.Tensor,
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_states_tensor: torch.Tensor,
        dones_tensor: torch.Tensor,
        weights_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the loss for the samples from the replay buffer.
        The loss is computed as the mean squared error between the current
        Q values and the target Q values.

        Parameters
        ----------
        states_tensor : torch.Tensor
            The states from the replay buffer.
        actions_tensor : torch.Tensor
            The actions from the replay buffer.
        rewards_tensor : torch.Tensor
            The rewards from the replay buffer.
        next_states_tensor : torch.Tensor
            The next states from the replay buffer.
        dones_tensor : torch.Tensor
            The dones from the replay buffer.
        weights_tensor : torch.Tensor
            The weights from the replay buffer.

        Returns
        -------
        torch.Tensor
            The loss for the samples from the replay buffer.
        """
        q_current: torch.Tensor = (
            self.q_network(states_tensor)
            .gather(1, actions_tensor.unsqueeze(1))
            .squeeze(1)
        )

        best_actions = self.q_network(next_states_tensor).detach().max(1)[1]

        q_next: torch.Tensor = self.target_network(next_states_tensor).detach()
        q_next_max = q_next.gather(1, best_actions.unsqueeze(1)).squeeze(1)

        q_targets: torch.Tensor = rewards_tensor + (
            self.gamma * q_next_max * (1 - dones_tensor)
        )

        loss: torch.Tensor = (q_current - q_targets) ** 2 * weights_tensor
        return loss.mean(), q_targets

    def calculate_q_value_diff(self, state: np.ndarray) -> None:
        """
        Calculate the Q-value difference between the target and local network.

        Parameters
        ----------
        states : np.ndarray
            The state from the replay buffer.
        """
        state_tensor = torch.from_numpy(np.array([state], dtype=np.float32)).to(device)
        with torch.no_grad():
            current_q_values = self.q_network(state_tensor)
            current_average_q_value = current_q_values.mean().item()

        if self.previous_q_value is not None:
            diff = abs(current_average_q_value - self.previous_q_value)
            self.q_value_diff.append(diff)

        self.previous_q_value = current_average_q_value

    def compute_td_error_from_experience(self, state, action, reward, next_state, done):
        """
        Compute TD error from a single experience.

        Parameters
        ----------
        state : np.ndarray
            The state from the replay buffer.
        action : int
            The action from the replay buffer.
        reward : float
            The reward from the replay buffer.
        next_state : np.ndarray
            The next state from the replay buffer.
        done : bool
            The done from the replay buffer.

        Returns
        -------
        td_error : float
            The TD error for the single experience.
        """
        state_tensor = torch.from_numpy(np.array([state])).float().to(device)
        next_state_tensor = torch.from_numpy(np.array([next_state])).float().to(device)
        action_tensor = torch.from_numpy(np.array([action])).long().to(device)
        reward_tensor = torch.from_numpy(np.array([reward])).float().to(device)
        done_tensor = torch.from_numpy(np.array([done])).long().to(device)

        # Compute Q-values for current states and actions
        q_current = (
            self.q_network(state_tensor)
            .gather(1, action_tensor.unsqueeze(1))
            .squeeze(1)
        )

        # Select best actions for next states from the Q-network
        best_actions = self.q_network(next_state_tensor).detach().max(1)[1]

        # Compute Q-values for next states
        q_next = self.target_network(next_state_tensor).detach()
        q_next_max = q_next.gather(1, best_actions.unsqueeze(1)).squeeze(1)

        # Compute expected Q-values
        q_targets = reward_tensor + (self.gamma * q_next_max * (1 - done_tensor))

        # Compute TD error
        td_error = q_current - q_targets

        return td_error.item()

    def update_priorities(self, indices: np.ndarray, errors: np.ndarray) -> None:
        """
        Update the priorities of the replay buffer.

        Parameters
        ----------
        indices : np.ndarray
            The indices of the samples.
        errors : np.ndarray
            The errors for the samples.
        """
        self.memory.update_priorities(indices, errors)

    def update_target_network(self):
        """
        Soft update of the target network's weights.
        """
        for target_param, local_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def update_epsilon(self):
        """
        Decay epsilon used for epsilon-greedy policy.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_q_value(self, state: np.ndarray):
        """
        Get the Q-value for the current state.

        Parameters
        ----------
        state : np.ndarray
            The state from the replay buffer.

        Returns
        -------
        float
            The Q-value for the current state.
        """
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
        return self.q_network(state).detach().cpu().numpy()

    def get_learning_rate(self):
        """
        Get the learning rate for the current state.

        Returns
        -------
        float
            The learning rate for the current state.
        """
        return self.scheduler.get_last_lr()[0]

    def get_epsilon(self):
        """
        Get the epsilon for the current state.

        Returns
        -------
        float
            The epsilon for the current state.
        """
        return self.epsilon

    def log_metrics(
        self,
        episode: int,
        reward: float,
        epsilon: float,
        lr: float,
        q_values: np.ndarray,
        action_distribution: np.ndarray,
        td_error: float,
        loss: float,
        average_q_value_diff: float,
        episode_length: int,
    ):
        """
        Log metrics to TensorBoard.

        Parameters
        ----------
        episode : int
            The current episode.
        reward : float
            The current reward.
        epsilon : float
            The current epsilon.
        lr : float
            The current learning rate.
        q_values : np.array
            The current Q-values.
        action_distribution : np.array
            The current action distribution.
        td_error : float
            The current TD error.
        loss : float
            The current loss.
        average_q_value_diff : float
            The average Q-value difference.
        """
        self.writer.add_scalar("Reward/episode", reward, episode)
        self.writer.add_scalar("Epsilon/episode", epsilon, episode)
        self.writer.add_scalar("Learning Rate/episode", lr, episode)
        self.writer.add_histogram("Q Values/episode", q_values, episode)
        self.writer.add_histogram(
            "Action Distribution/episode", action_distribution, episode
        )
        self.writer.add_scalar("TD Error/episode", td_error, episode)
        self.writer.add_scalar("Loss/episode", loss, episode)
        self.writer.add_histogram(
            "Q Value Differences/episode", average_q_value_diff, episode
        )
        self.writer.add_scalar("Episode Length/episode", episode_length, episode)

    def save_model(self, filename: str):
        """
        Save the model.

        Parameters
        ----------
        filename : str
            The filename to save the model to.
        """
        torch.save(self.q_network.state_dict(), filename)
        self.writer.close()

    def load_model(self, filename: str):
        """
        Load the model.

        Parameters
        ----------
        filename : str
            The filename to load the model from.
        """
        self.q_network.load_state_dict(torch.load(filename))
        self.q_network.eval()

        self.target_network.load_state_dict(torch.load(filename))
        self.target_network.eval()

    def __del__(self):
        self.writer.close()
