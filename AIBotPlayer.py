from Player import Player
from Models import PigModel
from Predictor import Predictor
import torch
import random

class AIBotPlayer(Player):
    # STRATEGY = ("this bot goes for 23 because thats what i tested to be good value. "
    #             "however, it changes based on how close the bot is to winning, and how close opponents are to winning")
    
    
    def __init__(self, name, model, train, model_path = "", device='cpu', strategy=None):
        super().__init__(name)
        if strategy is not None:
            self.STRATEGY = strategy
        self.model = model.to(device)
        self.train = train
        self.device = device
        if not train:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()

    # def wants_to_roll(self, my_score, hand_score, other_scores, winning_score):
    #     input_tuple = (my_score, hand_score, self.most_dangerous_opponent_proximity(other_scores, winning_score), winning_score)
    #     input_tensor = torch.tensor(input_tuple, dtype=torch.float32)
    #     input_tensor = input_tensor.unsqueeze(0)
    #     self.model.eval()
    #     random_number = random.random()
    #     output = self.model(input_tensor)
    #     probability = output[0, 0].item()
    #     if random_number <= probability:
    #         return True, probability
    #     else:
    #         return False, 1 - probability

    def wants_to_roll(self, my_score, hand_score, other_scores, winning_score):
        # Build the input tensor with the required features.
        
        input_tuple = (
            my_score, 
            hand_score, 
            self.most_dangerous_opponent_proximity(other_scores, winning_score),
            winning_score
        )
        input_tensor = torch.tensor(input_tuple, dtype=torch.float32).unsqueeze(0)
        
        # Set the model to evaluation mode and compute the output.
        self.model.eval()
        output = self.model(input_tensor)
        # Assume the model outputs a tensor of shape [1, 2], where index 0 is the probability for "roll".
        probability = output[0, 0]  # Keep as tensor so gradients can flow if needed.
        
        # Sample an action using the probability.
        probability = torch.where(torch.isnan(probability), torch.tensor(0.5, device=probability.device), probability)
        probability = probability.clamp(1e-6, 1 - 1e-6)
        action_sample = torch.bernoulli(probability)

        # Always return a tuple: (decision, probability)
        if action_sample.item() == 1:
            return (True, probability)
        else:
            return (False, 1 - probability)


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
