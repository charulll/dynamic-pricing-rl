from environment import DynamicPricingEnv
from q_learning import QLearningAgent

env = DynamicPricingEnv()

agent = QLearningAgent(
    num_actions=env.action_space.n
)

episodes = 1000

revenues = []

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

avg_revenue = sum(revenues) / len(revenues)

print("\nTraining Complete")
print(
    f"Average Revenue across {episodes} episodes: "
    f"{avg_revenue:.2f}"
)