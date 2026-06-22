from src.environment import DynamicPricingEnv
from baseline_agents import FixedPriceAgent, DiscountAgent
from src.q_learning import QLearningAgent
import pickle


def evaluate_agent(agent, env, episodes=100):

    revenues = []

    for _ in range(episodes):

        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:

            action = agent.choose_action(state)

            next_state, reward, done, _, _ = env.step(action)

            state = next_state
            total_reward += reward

        revenues.append(total_reward)

    return sum(revenues) / len(revenues)


env = DynamicPricingEnv()

# Fixed Agent
fixed_agent = FixedPriceAgent()
fixed_avg = evaluate_agent(fixed_agent, env)

# Discount Agent
discount_agent = DiscountAgent()
discount_avg = evaluate_agent(discount_agent, env)

# Q-Learning Agent
q_agent = QLearningAgent(num_actions=env.action_space.n)

with open("q_table.pkl", "rb") as f:
    q_agent.q_table.update(pickle.load(f))

q_agent.epsilon = 0

q_avg = evaluate_agent(q_agent, env)

print(f"Fixed Agent Avg Revenue: {fixed_avg:.2f}")
print(f"Discount Agent Avg Revenue: {discount_avg:.2f}")
print(f"Q-Learning Avg Revenue: {q_avg:.2f}")