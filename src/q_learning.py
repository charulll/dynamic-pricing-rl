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

    def choose_action(self, state):

        state = tuple(state.astype(int))

        if np.random.random() < self.epsilon:
            return np.random.randint(
                self.num_actions
            )

        return np.argmax(
            self.q_table[state]
        )

    def update(
        self,
        state,
        action,
        reward,
        next_state
    ):

        state = tuple(state.astype(int))
        next_state = tuple(
            next_state.astype(int)
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