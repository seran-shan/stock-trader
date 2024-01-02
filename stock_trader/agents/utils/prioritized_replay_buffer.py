"""
This file contains the Prioritized Replay Buffer class, 
used for prioritized replay in the DDQN agent.
"""
import numpy as np

# pylint: disable=import-error
from agents.utils.sum_tree import SumTree


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
