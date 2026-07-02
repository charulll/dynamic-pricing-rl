import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DynamicPricingEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.max_inventory = 100
        self.max_days = 30

        self.prices = [80, 100, 120, 140, 160]

        self.action_space = spaces.Discrete(len(self.prices))

        self.observation_space = spaces.Box(
        low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        dtype=np.float32,
    )

        self.reset()

    # ---------------------------------------------------------
    # Demand Simulation
    # ---------------------------------------------------------
    def get_demand(self, price):

        # Daily market demand fluctuates
        market_factor = np.random.uniform(0.8, 1.2)

        # Customers arriving today
        customers_today = int(np.random.poisson(18) * market_factor)

        sales = 0

        for _ in range(customers_today):

            # 30% Business Travellers
            if np.random.rand() < 0.30:

                # Business travellers mostly book late
                if self.days_left <= 10:

                    if price <= 160:
                        buy_prob = 0.90

                else:

                    if price <= 140:
                        buy_prob = 0.70
                    else:
                        buy_prob = 0.50

            # 70% Leisure Travellers
            else:

                # Early booking
                if self.days_left > 20:

                    if price == 80:
                        buy_prob = 0.95
                    elif price == 100:
                        buy_prob = 0.90
                    elif price == 120:
                        buy_prob = 0.75
                    elif price == 140:
                        buy_prob = 0.50
                    else:
                        buy_prob = 0.25

                # Middle booking period
                elif self.days_left > 10:

                    if price == 80:
                        buy_prob = 0.85
                    elif price == 100:
                        buy_prob = 0.75
                    elif price == 120:
                        buy_prob = 0.60
                    elif price == 140:
                        buy_prob = 0.40
                    else:
                        buy_prob = 0.15

                # Last minute
                else:

                    if price == 80:
                        buy_prob = 0.95
                    elif price == 100:
                        buy_prob = 0.85
                    elif price == 120:
                        buy_prob = 0.55
                    elif price == 140:
                        buy_prob = 0.20
                    else:
                        buy_prob = 0.05

            # ---------------- Inventory Effect ----------------

            inventory_ratio = self.inventory / self.max_inventory

            # Too many rooms left near departure
            if inventory_ratio > 0.70 and self.days_left < 8:

                if price <= 100:
                    buy_prob += 0.20

            # Almost sold out
            elif inventory_ratio < 0.20:

                if price >= 140:
                    buy_prob += 0.20

            buy_prob = np.clip(buy_prob, 0.02, 0.98)

            if np.random.rand() < buy_prob:
                sales += 1

        return sales

    # ---------------------------------------------------------
    # Reset
    # ---------------------------------------------------------

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.inventory = self.max_inventory
        self.days_left = self.max_days
        # Initial market demand
        self.market_demand = np.random.uniform(0.6, 1.0)

        state = np.array(
            [
                self.inventory / self.max_inventory,
                self.days_left / self.max_days,
                self.market_demand,
            ],
            dtype=np.float32,
        )

        return state, {}

    # ---------------------------------------------------------
    # Step
    # ---------------------------------------------------------

    def step(self, action):

        price = self.prices[action]

        sales = self.get_demand(price)

        sales = min(sales, self.inventory)

        revenue = sales * price

        reward = revenue

        # Bonus for clearing inventory near deadline
        if self.days_left <= 5:
            reward += sales * 20

        self.inventory -= sales
        self.days_left -= 1
        # Market demand changes every day
        self.market_demand += np.random.normal(0, 0.05)
        self.market_demand = np.clip(self.market_demand, 0.2, 1.0)

        done = (
            self.inventory <= 0
            or self.days_left <= 0
        )

        # Penalty for unsold inventory
        if done and self.inventory > 0:
            reward -= self.inventory * 80

        # Bonus if sold out before departure
        if done and self.inventory == 0:
            reward += 1000

        next_state = np.array(
            [
                self.inventory / self.max_inventory,
                self.days_left / self.max_days,
                self.market_demand,
            ],
            dtype=np.float32,
        )

        return (
            next_state,
            reward,
            done,
            False,
            {},
        )

    # ---------------------------------------------------------
    # Render
    # ---------------------------------------------------------

    def render(self):

        print(
            f"Inventory: {self.inventory}, "
            f"Days Left: {self.days_left}"
        )