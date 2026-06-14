class FixedPriceAgent:
    def choose_action(self, state):
        return 2


class DiscountAgent:
    def choose_action(self, state):
        inventory, days_left = state

        if days_left > 7:
            return 4
        elif days_left > 3:
            return 3
        else:
            return 1