from PassThePigs import PassThePigs
from Linear23Bot import Linear23Bot
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Predictor import Predictor
import random
from Trainer import PigTrainer
from AIBotPlayer import AIBotPlayer

WINNING_SCORE = 100
average = 0.0
print_game = False

def one_player_turn(player_number, players, piggy):
    global average, print_game
    pigged_out = False
    passed = False

    while not pigged_out and not passed:
        player_wants_to_role, probobility =  players[player_number].wants_to_roll(
            piggy.get_player_bank(player_number),
            piggy.get_hand_value(),
            piggy.get_players_bank_values(player_number),
            WINNING_SCORE
        )
        if player_wants_to_role:
            if print_game:
                print(f"{players[player_number].get_name()} rolls a ", end="")
            pigged_out = piggy.player_role_pigs()
        else:
            if print_game:
                print(f"{players[player_number].get_name()} passes")
            average += 1.0 / 1000000.0
            passed = True

    piggy.change_player_bank_after_round(player_number)

def display_game_status(players, piggy):
    if print_game:
        for i, player in enumerate(players):
            print(f"{player.get_name()}: {piggy.get_player_bank(i)} | ", end="")
        print()
        
def reset(piggy, amount_of_players):
    piggy.reset()
    random_float_range = random.uniform(0, amount_of_players - 1)
    player_number = random_float_range
    won = False
    return player_number, won

def main():
    train = True
    trainer = PigTrainer(2)
    if train:
        for _ in range(10):
            trainer.train()
            trainer.save_model("./models/model.pth")
    else:    
        global print_game
        won = False
        player_number = 0

        players = []
        # Uncomment or add additional players as needed:
        # players.append(HumanPlayer("tony"))
        # players.append(TestBot("testBot"))
        #players.append(RiskyBotPlayerShawn("riskyBot"))
        players.append(AIBotPlayer("linear23Bot"))
        players.append(Linear23Bot("exponentialDecayBot"))
        # players.append(TestBot("testBot2"))
        # players.append(RiskyBotPlayer("riskyBot2"))
        # players.append(BestBot("bestBot2"))
        # players.append(ExponentialDecayBot("exponentialDecayBot2"))

        piggy = PassThePigs(len(players), print_game)

        while not won:
            one_player_turn(player_number, players, piggy)
            display_game_status(players, piggy)

            if piggy.get_player_bank(player_number) >= WINNING_SCORE:
                won = True
                if print_game:
                    print("Game over! Winner is: " + players[player_number].get_name())
            else:
                player_number = (player_number + 1) % len(players)


if __name__ == "__main__":
    main()
