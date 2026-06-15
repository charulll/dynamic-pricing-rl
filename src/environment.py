import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DynamicPricingEnv(gym.Env):

    def __init__(self):

        super().__init__()

        self.max_inventory = 100
        self.max_days = 30

        # 5 possible price levels
        self.prices = [80, 100, 120, 140, 160]

        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([100, 30]),
            dtype=np.float32
        )

        self.reset()

    def get_demand(self, price):

    # Random customers arriving today
        customers_today = np.random.poisson(15)

    # Purchase probability decreases with price
    # and increases as departure date approaches
        purchase_prob = (
             0.8
            - (price / 250)
            + ((self.max_days - self.days_left) / 100)
        )

    # Keep probability within reasonable limits
        purchase_prob = np.clip(
            purchase_prob,
            0.05,
            0.95
        )

        sales = 0

        for _ in range(customers_today):

            if np.random.random() < purchase_prob:
                sales += 1

        return sales

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.inventory = self.max_inventory
        self.days_left = self.max_days

        state = np.array(
            [self.inventory, self.days_left],
            dtype=np.float32
        )

        return state, {}

    def step(self, action):

        price = self.prices[action]

        sales = self.get_demand(price)

        sales = min(sales, self.inventory)

        reward = sales * price

        self.inventory -= sales
        self.days_left -= 1

        done = (
            self.inventory <= 0
            or self.days_left <= 0
        )

        next_state = np.array(
            [self.inventory, self.days_left],
            dtype=np.float32
        )

        return (
            next_state,
            reward,
            done,
            False,
            {}
        )

    def render(self):

        print(
            f"Inventory: {self.inventory}, "
            f"Days Left: {self.days_left}"
        )