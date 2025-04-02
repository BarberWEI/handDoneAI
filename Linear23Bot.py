from BotPlayer import BotPlayer

class Linear23Bot(BotPlayer):
    STRATEGY = "this bot goes for 23 as the target value, but when opponents are close to winning, it goes for higher values"

    def __init__(self, name):
        super().__init__(name, Linear23Bot.STRATEGY)

    def wants_to_roll(self, my_score, hand_score, other_scores, winning_score):
        # Through trial and error I got 32 as the quickest way to 100 on average of 10,000 tests
        role = True
        if winning_score - my_score < 40:
            if hand_score >= winning_score - my_score:
                role = False
        else:
            role = self._if_not_near_winning(other_scores, winning_score, hand_score)
        return (role, 1)

    def _if_not_near_winning(self, other_scores, winning_score, hand_score):
        opponent_closest_to_winning = self.most_dangerous_opponent_proximity(other_scores, winning_score)
        distance_from_player = winning_score - opponent_closest_to_winning - hand_score

        if distance_from_player > 40:
            if hand_score >= 45 or winning_score - hand_score <= 0:
                return False
        elif distance_from_player > 35:
            if hand_score >= 40 or winning_score - hand_score <= 0:
                return False
        elif distance_from_player > 30:
            if hand_score >= 35 or winning_score - hand_score <= 0:
                return False
        elif distance_from_player > 25:
            if hand_score >= 30 or winning_score - hand_score <= 0:
                return False
        elif distance_from_player > 20:
            if hand_score >= 25 or winning_score - hand_score <= 0:
                return False
        elif hand_score >= 23:
            return False
        return True

    def get_strategy(self):
        return Linear23Bot.STRATEGY
