"""
train_dqn.py — Week 3 DQN Training Script
==========================================
Run from inside the src/ directory:

    cd src
    python train_dqn.py

Mirrors the structure of train_qlearning.py so both scripts are
interchangeable for comparison purposes.
"""

import sys
import os
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup — allow bare imports identical to train_qlearning.py,
# while also supporting execution from the project root.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))   # src/ on the path

from environment import DynamicPricingEnv   # Week 1
from dqn_agent import DQNAgent             # Week 3

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
EPISODES          = 500    # total training episodes
BATCH_SIZE        = 64     # minibatch size for each learn() call
BUFFER_CAPACITY   = 10_000 # replay buffer max transitions
LR                = 1e-3   # Adam learning rate
GAMMA             = 0.95   # discount factor  (matches Q-Learning)
EPSILON_START     = 1.0    # start fully random
EPSILON_MIN       = 0.01   # floor on exploration
EPSILON_DECAY     = 0.995  # multiplicative decay per learn() call
TARGET_UPDATE_FREQ = 10    # sync target net every N episodes
PRINT_EVERY       = 50     # console log interval
SAVE_PATH         = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "..", "results", "dqn_weights.pth"
)

# ---------------------------------------------------------------------------
# Environment + Agent
# ---------------------------------------------------------------------------
env = DynamicPricingEnv()

agent = DQNAgent(
    state_dim         = env.observation_space.shape[0],  # 2
    num_actions       = env.action_space.n,              # 5
    lr                = LR,
    gamma             = GAMMA,
    epsilon           = EPSILON_START,
    epsilon_min       = EPSILON_MIN,
    epsilon_decay     = EPSILON_DECAY,
    buffer_capacity   = BUFFER_CAPACITY,
    batch_size        = BATCH_SIZE,
    target_update_freq = TARGET_UPDATE_FREQ,
)

print(f"Training DQN for {EPISODES} episodes on device: {agent.device}\n")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
revenues = []   # total revenue collected per episode

for episode in range(1, EPISODES + 1):

    state, _ = env.reset()
    done = False
    total_revenue = 0.0

    # ---- Collect transitions for one episode ----------------------------
    while not done:

        # Choose action (epsilon-greedy during training)
        action = agent.choose_action(state)

        # Step the environment
        next_state, reward, done, _, _ = env.step(action)

        # Store transition in replay buffer
        agent.store(state, action, reward, next_state, done)

        # One gradient-descent step (returns None if buffer not full yet)
        agent.learn()

        state = next_state
        total_revenue += reward

    revenues.append(total_revenue)

    # ---- Periodic target-network sync -----------------------------------
    if episode % TARGET_UPDATE_FREQ == 0:
        agent.update_target()

    # ---- Console progress -----------------------------------------------
    if episode % PRINT_EVERY == 0:
        avg = sum(revenues[-PRINT_EVERY:]) / PRINT_EVERY
        print(
            f"Episode {episode:>4}/{EPISODES} | "
            f"Revenue: {total_revenue:>8.0f} | "
            f"Avg (last {PRINT_EVERY}): {avg:>8.0f} | "
            f"Epsilon: {agent.epsilon:.4f}"
        )

# ---------------------------------------------------------------------------
# Final stats
# ---------------------------------------------------------------------------
overall_avg = sum(revenues) / len(revenues)
print("\nTraining Complete")
print(f"Average Revenue across {EPISODES} episodes: {overall_avg:.2f}")

# ---------------------------------------------------------------------------
# Save model weights
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
agent.save(SAVE_PATH)
print(f"Model weights saved → {SAVE_PATH}")

# ---------------------------------------------------------------------------
# Revenue curve
# ---------------------------------------------------------------------------
# Smooth the raw curve with a rolling window for readability (O(n) pass)
WINDOW = 20
rolling_avg = []
running_sum = 0.0

for i, rev in enumerate(revenues):
    running_sum += rev
    if i >= WINDOW:
        running_sum -= revenues[i - WINDOW]
    window_len = min(i + 1, WINDOW)
    rolling_avg.append(running_sum / window_len)

plt.figure(figsize=(10, 5))
plt.plot(revenues, alpha=0.3, color="steelblue", label="Episode Revenue")
plt.plot(rolling_avg, color="steelblue", linewidth=2,
         label=f"Rolling Avg (window={WINDOW})")
plt.xlabel("Episode")
plt.ylabel("Total Revenue")
plt.title("DQN Training — Revenue Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
