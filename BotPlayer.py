from Player import Player

class BotPlayer(Player):
    STRATEGY = ("this bot goes for 23 because thats what i tested to be good value. "
                "however, it changes based on how close the bot is to winning, and how close opponents are to winning")

    def __init__(self, name, strategy=None):
        super().__init__(name)
        if strategy is not None:
            self.STRATEGY = strategy

    def wants_to_roll(self, my_score, hand_score, other_scores, winning_score):
        role = True
        return role

    def _if_not_near_winning(self, other_scores, winning_score, hand_score):
        return True

    def most_dangerous_opponent_proximity(self, other_scores, winning_score):
        if len(other_scores) == 0:
            return 1
        min_difference = winning_score - other_scores[0]
        for score in other_scores:
            if winning_score - score < min_difference:
                min_difference = winning_score - score
        return min_difference

    def get_strategy(self):
        return self.STRATEGY
