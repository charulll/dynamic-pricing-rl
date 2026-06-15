from environment import DynamicPricingEnv

env = DynamicPricingEnv()

state, _ = env.reset()

for step in range(10):

    action = env.action_space.sample()

    next_state, reward, done, _, _ = env.step(action)

    print(
        f"Step {step}"
    )

    print(
        f"State: {next_state}"
    )

    print(
        f"Reward: {reward}"
    )

    print("-" * 30)

    if done:
        break