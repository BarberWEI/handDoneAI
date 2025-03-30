class Player:
    def __init__(self, name, strategy=None, risk_factor=0):
        self.name = name
        self.strategy = strategy
        self.risk_factor = risk_factor
        # In Java, a Scanner is used for input.
        # In Python, you can use input() when needed.

    def wants_to_roll(self, my_score, hand_score, other_scores, winning_score):
        role = False
        # Example decision logic (currently commented out):
        # if hand_score >= 22:
        #     role = False
        return role

    def get_name(self):
        return self.name

    def get_strategy(self):
        return self.strategy
