import torch

from src.environment import DynamicPricingEnv
from src.dqn_agent import DQNAgent

env = DynamicPricingEnv()

agent = DQNAgent(
    state_dim=env.observation_space.shape[0],
    num_actions=env.action_space.n,
)

agent.load("results/dqn_weights.pth")
agent.epsilon = 0

states = [
    [1.0, 1.0],      # full inventory, first day
    [0.8, 0.8],
    [0.5, 0.5],
    [0.2, 0.2],
    [0.1, 0.1],
]

for state in states:

    s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        q = agent.policy_net(s)

    print("\nState:", state)
    print("Q-values:", q.numpy())
    print("Best Action:", q.argmax().item())