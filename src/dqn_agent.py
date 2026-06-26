import random
import torch
import torch.nn as nn
from collections import deque, namedtuple


class QNetwork(nn.Module):
    """
    Feedforward Q-Network for the DynamicPricingEnv.

    Architecture:
        Input  (2)  ->  Linear(64)  ->  ReLU
                    ->  Linear(64)  ->  ReLU
                    ->  Linear(5)   ->  Q-values

    Args:
        state_dim   (int): Dimensionality of the state vector.
                           Matches the environment observation space (2).
        num_actions (int): Number of discrete actions.
                           Matches the environment action space (5).
    """

    def __init__(self, state_dim: int = 2, num_actions: int = 5):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): State tensor of shape (batch_size, state_dim)
                              or (state_dim,) for a single state.

        Returns:
            torch.Tensor: Q-value estimates of shape (batch_size, num_actions).
        """

        return self.network(x)


# Named tuple for storing individual transitions cleanly
Transition = namedtuple(
    "Transition",
    ["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    """
    Fixed-capacity experience replay buffer backed by a collections.deque.

    When the buffer is full, the oldest transition is automatically evicted
    to make room for the new one (deque maxlen behaviour).

    Args:
        capacity (int): Maximum number of transitions to store.
    """

    def __init__(self, capacity: int):

        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        done: bool,
    ) -> None:
        """
        Save a single transition to the buffer.

        Args:
            state:      np.ndarray — current environment state.
            action:     int        — action taken.
            reward:     float      — reward received.
            next_state: np.ndarray — resulting environment state.
            done:       bool       — whether the episode ended.
        """

        self.buffer.append(
            Transition(state, action, reward, next_state, done)
        )

    def sample(self, batch_size: int):
        """
        Randomly sample a batch of transitions without replacement.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            Transition: A namedtuple whose fields are each a list of
                        length batch_size, ready for conversion to tensors.

        Raises:
            ValueError: If batch_size exceeds the current buffer length.
        """

        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from a buffer "
                f"of size {len(self.buffer)}."
            )

        transitions = random.sample(self.buffer, batch_size)

        # Unzip list-of-Transitions into a single Transition-of-lists
        return Transition(*zip(*transitions))

    def __len__(self) -> int:
        """Return the current number of stored transitions."""

        return len(self.buffer)
