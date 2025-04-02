from Player import Player

class BotKaden(Player):
    def __init__(self, name):
        super().__init__(name)
        self.strategy = ""

    def wants_to_roll(self, my_score, hand_score, other_scores, winning_score):
        return (True, 0)

    def set_strategy(self, strat):
        self.strategy = strat
