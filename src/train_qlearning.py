from environment import DynamicPricingEnv
from q_learning import QLearningAgent
import matplotlib.pyplot as plt

# Create environment
env = DynamicPricingEnv()

# Create Q-Learning agent
agent = QLearningAgent(
    num_actions=env.action_space.n
)

# Number of training episodes
episodes = 1000

# Store revenue from each episode
revenues = []

# Training loop
for episode in range(episodes):

    state, _ = env.reset()

    done = False
    total_reward = 0

    while not done:

        action = agent.choose_action(state)

        next_state, reward, done, _, _ = env.step(action)

        agent.update(
            state,
            action,
            reward,
            next_state
        )

        state = next_state

        total_reward += reward

    revenues.append(total_reward)

    if (episode + 1) % 100 == 0:

        print(
            f"Episode {episode + 1}, Revenue: {total_reward}"
        )

# Average Revenue
avg_revenue = sum(revenues) / len(revenues)

print("\nTraining Complete")
print(
    f"Average Revenue across {episodes} episodes: "
    f"{avg_revenue:.2f}"
)

# Plot Revenue Trend
plt.figure(figsize=(10, 5))
plt.plot(revenues)

plt.xlabel("Episode")
plt.ylabel("Revenue")
plt.title("Q-Learning Revenue Trend")

plt.grid(True)

plt.show()