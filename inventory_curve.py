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
inventory = []

day = 0

while not done:

    # Store inventory before taking action
    days.append(day)
    inventory.append(env.inventory)

    action = agent.choose_action(state)

    state, reward, done, _, _ = env.step(action)

    day += 1

# Plot graph
plt.figure(figsize=(8,5))

plt.plot(days, inventory, marker="o", color="green")

plt.title("Inventory Remaining Over Time")

plt.xlabel("Day")

plt.ylabel("Remaining Inventory")

plt.grid(True)

plt.savefig("results/inventory_curve.png")

plt.close()

print("Inventory curve saved to results/inventory_curve.png")