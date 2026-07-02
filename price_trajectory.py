import matplotlib.pyplot as plt

from src.environment import DynamicPricingEnv
from src.dqn_agent import DQNAgent

# Create environment
env = DynamicPricingEnv()

# Create DQN agent
agent = DQNAgent(
    state_dim=env.observation_space.shape[0],
    num_actions=env.action_space.n,
)

# Load trained model
agent.load("results/dqn_weights.pth")

# Disable exploration
agent.epsilon = 0

# Start one episode
state, _ = env.reset()

done = False

days = []
prices = []

day = 0

while not done:

    action = agent.choose_action(state)

    # Convert action index to actual price
    price = env.prices[action]

    days.append(day)
    prices.append(price)

    state, reward, done, _, _ = env.step(action)

    day += 1

# Plot graph
plt.figure(figsize=(8,5))

plt.plot(days, prices, marker="o")

plt.title("Price Trajectory of DQN Agent")

plt.xlabel("Day")

plt.ylabel("Selected Price")

plt.grid(True)

plt.savefig("results/price_trajectory.png")

plt.close()

print("Price trajectory saved to results/price_trajectory.png")