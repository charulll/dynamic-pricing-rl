from src.environment import DynamicPricingEnv
from baseline_agents import FixedPriceAgent, DiscountAgent

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

fixed_agent = FixedPriceAgent()
discount_agent = DiscountAgent()

fixed_avg = evaluate_agent(fixed_agent, env)
discount_avg = evaluate_agent(discount_agent, env)

print(f"Fixed Agent Avg Revenue: {fixed_avg:.2f}")
print(f"Discount Agent Avg Revenue: {discount_avg:.2f}")