import torch
import torch.nn as nn


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
