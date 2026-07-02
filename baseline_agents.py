class FixedPriceAgent:

    def choose_action(self, state):

        return 2     # Always ₹120


class DiscountAgent:

    def choose_action(self, state):

        inventory, days_left, demand = state

        if days_left > 0.7:
            return 4      # ₹160

        elif days_left > 0.4:
            return 3      # ₹140

        elif days_left > 0.2:
            return 2      # ₹120

        else:
            return 1      # ₹100