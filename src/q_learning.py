import numpy as np
from collections import defaultdict


class QLearningAgent:

    def __init__(
        self,
        num_actions,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.1
    ):

        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table = defaultdict(
            lambda: np.zeros(num_actions)
        )

    # ----------------------------
    # Convert continuous state into
    # discrete buckets
    # ----------------------------
    def discretize_state(self, state):

        inventory = int(state[0] * 10)
        days = int(state[1] * 10)
        demand = int(state[2] * 10)

        inventory = min(inventory, 9)
        days = min(days, 9)
        demand = min(demand, 9)

        return (
            inventory,
            days,
            demand
        )

    # ----------------------------

    def choose_action(self, state):

        state = self.discretize_state(state)

        if np.random.random() < self.epsilon:

            return np.random.randint(
                self.num_actions
            )

        return np.argmax(
            self.q_table[state]
        )

    # ----------------------------

    def update(
        self,
        state,
        action,
        reward,
        next_state
    ):

        state = self.discretize_state(state)

        next_state = self.discretize_state(
            next_state
        )

        best_next_action = np.argmax(
            self.q_table[next_state]
        )

        td_target = (
            reward
            + self.gamma *
            self.q_table[next_state][best_next_action]
        )

        td_error = (
            td_target -
            self.q_table[state][action]
        )

        self.q_table[state][action] += (
            self.alpha * td_error
        )