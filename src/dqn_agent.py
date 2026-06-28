"""
dqn_agent.py — Week 3: Deep Q-Network components
=================================================
Contains three classes built on top of the DynamicPricingEnv (Week 1):

    QNetwork     — PyTorch neural network mapping state -> Q-values.
    ReplayBuffer — Fixed-capacity experience replay buffer.
    DQNAgent     — Full DQN agent satisfying the evaluate_agent() contract
                   defined in compare_agents.py.
"""

import copy
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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


class DQNAgent:
    """
    Deep Q-Network agent for the DynamicPricingEnv.

    Maintains two networks (policy + target) to stabilise training,
    an experience replay buffer, and an epsilon-greedy exploration schedule.

    The public interface satisfies the existing evaluate_agent() contract:
        agent.choose_action(state: np.ndarray) -> int

    Args:
        state_dim         (int):   Size of the state vector (2).
        num_actions       (int):   Number of discrete actions (5).
        lr                (float): Adam learning rate.
        gamma             (float): Discount factor — matches Q-Learning default.
        epsilon           (float): Starting exploration rate.
        epsilon_min       (float): Lowest exploration rate allowed.
        epsilon_decay     (float): Multiplicative decay applied after each step.
        buffer_capacity   (int):   Max transitions stored in replay buffer.
        batch_size        (int):   Minibatch size drawn per training step.
        target_update_freq(int):   Episodes between hard target-network syncs.
    """

    def __init__(
        self,
        state_dim: int = 2,
        num_actions: int = 5,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 10,
    ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Use GPU if available, otherwise fall back to CPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Policy network: updated every training step via backprop
        self.policy_net = QNetwork(state_dim, num_actions).to(self.device)

        # Target network: frozen copy of policy_net, synced periodically
        # Using copy.deepcopy ensures weights are independent at init
        self.target_net = copy.deepcopy(self.policy_net).to(self.device)
        self.target_net.eval()   # target net is never put in training mode

        # Adam optimizer — only updates policy_net parameters
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Huber loss is less sensitive to large reward outliers than MSE
        self.loss_fn = nn.HuberLoss()

        # Experience replay buffer
        self.buffer = ReplayBuffer(capacity=buffer_capacity)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def choose_action(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy action selection.

        During training, explores randomly with probability epsilon;
        otherwise acts greedily on policy_net Q-values.
        Setting self.epsilon = 0 before evaluation gives pure-greedy
        behaviour, which is what compare_agents.py requires.

        Args:
            state (np.ndarray): Current environment state of shape (2,).

        Returns:
            int: Selected action index in [0, num_actions).
        """

        # Random exploration
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        # Greedy: convert state to tensor, forward through policy net
        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)   # shape: (1, state_dim)

        with torch.no_grad():
            q_values = self.policy_net(state_t)   # shape: (1, num_actions)

        return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Replay buffer interaction
    # ------------------------------------------------------------------

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Push one transition into the replay buffer.
        Called once per environment step during the training loop.

        Args:
            state:      np.ndarray — current environment state.
            action:     int        — action taken.
            reward:     float      — reward received.
            next_state: np.ndarray — resulting environment state.
            done:       bool       — whether the episode ended.
        """

        self.buffer.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def learn(self) -> Optional[float]:
        """
        Sample a minibatch from the replay buffer and perform one
        gradient-descent step on the policy network.

        Returns:
            float | None: The scalar loss for this step, or None if the
                          buffer does not yet hold enough transitions.
        """

        # Do nothing until the buffer has at least one full minibatch
        if len(self.buffer) < self.batch_size:
            return None

        # ---- 1. Sample minibatch ----------------------------------------
        batch = self.buffer.sample(self.batch_size)

        # Stack numpy arrays / scalars into tensors on the right device
        states = torch.tensor(
            np.array(batch.state), dtype=torch.float32, device=self.device
        )                                               # (B, state_dim)
        actions = torch.tensor(
            batch.action, dtype=torch.long, device=self.device
        ).unsqueeze(1)                                  # (B, 1)
        rewards = torch.tensor(
            batch.reward, dtype=torch.float32, device=self.device
        ).unsqueeze(1)                                  # (B, 1)
        next_states = torch.tensor(
            np.array(batch.next_state), dtype=torch.float32, device=self.device
        )                                               # (B, state_dim)
        dones = torch.tensor(
            batch.done, dtype=torch.float32, device=self.device
        ).unsqueeze(1)                                  # (B, 1)  — 1.0 if terminal

        # ---- 2. Current Q-values from policy net ------------------------
        # Gather the Q-value for the action that was actually taken
        current_q = self.policy_net(states).gather(1, actions)  # (B, 1)

        # ---- 3. Target Q-values from target net -------------------------
        # max Q(s', a') over all next actions; zero out terminal states
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(
                dim=1, keepdim=True
            ).values                                    # (B, 1)
            target_q = rewards + self.gamma * max_next_q * (1.0 - dones)

        # ---- 4. Compute loss and backpropagate --------------------------
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping prevents exploding gradients with large rewards
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)

        self.optimizer.step()

        # ---- 5. Decay epsilon after each learning step ------------------
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    # ------------------------------------------------------------------
    # Target network synchronisation
    # ------------------------------------------------------------------

    def update_target(self) -> None:
        """
        Hard copy: overwrite target_net weights with policy_net weights.
        Should be called every `target_update_freq` episodes from the
        training loop. Keeps the TD target stable between syncs.
        """

        self.target_net.load_state_dict(self.policy_net.state_dict())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save policy network weights to disk.
        Only the policy net is saved; the target net is reconstructed
        from it via update_target() or deepcopy at load time.

        Args:
            path (str): File path, conventionally ending in '.pth'.
        """

        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        """
        Load policy network weights from disk and sync the target net.
        After loading, set self.epsilon = 0 for greedy evaluation.

        Args:
            path (str): File path to a saved '.pth' state dict.
        """

        self.policy_net.load_state_dict(
            # weights_only=True avoids the FutureWarning in PyTorch >= 2.0
            torch.load(path, map_location=self.device, weights_only=True)
        )
        # Keep the target net in sync with the loaded weights
        self.update_target()
