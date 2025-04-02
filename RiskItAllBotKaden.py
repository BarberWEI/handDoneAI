from Player import Player  # or whatever base class you're extending

class RiskItAllBotKaden(Player):
    def __init__(self, name):
        super().__init__(name)
        self.strategy = "Like GetAheadBot, but it really does not stop until it gets ahead."

    def wants_to_roll(self, my_score, hand_score, other_scores, winning_score):
        if hand_score < 15:
            return (True, 0)

        for score in other_scores:
            if score > hand_score:
                return (True, 0)

        return (False, 0)
