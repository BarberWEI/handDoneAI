from PIL import Image        
from PassThePigs import PassThePigs
from Linear23Bot import Linear23Bot
from Trainer import MnistTrainer
from torchvision import datasets, transforms
from Models import MnistModel
from Models import PigDecisionNet
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from Predictor import Predictor

WINNING_SCORE = 100
average = 0.0
print_game = True

def one_player_turn(player_number, players, piggy):
    global average, print_game
    pigged_out = False
    passed = False

    while not pigged_out and not passed:
        if players[player_number].wants_to_roll(
            piggy.get_player_bank(player_number),
            piggy.get_hand_value(),
            piggy.get_players_bank_values(player_number),
            WINNING_SCORE
        ):
            if print_game:
                print(f"{players[player_number].get_name()} rolls a ", end="")
            pigged_out = piggy.player_role_pigs(player_number)
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

def main():
    train = True
    
    if not train:
        predictor = Predictor(MnistModel, "models/mnist_model.pth", 784, [128, 64], 10)


        img = Image.open("path_to_your_image.png").convert("L")  # MNIST is grayscale
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # Example for MNIST
            transforms.Lambda(lambda x: x.view(-1))
        ])
        
        tensor = transform(img)

        # global print_game
        # won = False
        # player_number = 0

        # players = []
        # # Uncomment or add additional players as needed:
        # # players.append(HumanPlayer("tony"))
        # # players.append(TestBot("testBot"))
        # #players.append(RiskyBotPlayerShawn("riskyBot"))
        # players.append(Linear23Bot("linear23Bot"))
        # players.append(Linear23Bot("exponentialDecayBot"))
        # # players.append(TestBot("testBot2"))
        # # players.append(RiskyBotPlayer("riskyBot2"))
        # # players.append(BestBot("bestBot2"))
        # # players.append(ExponentialDecayBot("exponentialDecayBot2"))

        # piggy = PassThePigs(len(players), print_game)

        # while not won:
        #     one_player_turn(player_number, players, piggy)
        #     display_game_status(players, piggy)

        #     if piggy.get_player_bank(player_number) >= WINNING_SCORE:
        #         won = True
        #         if print_game:
        #             print("Game over! Winner is: " + players[player_number].get_name())
        #     else:
        #         player_number = (player_number + 1) % len(players)
    else:
        
        model = MnnistModel(784, [128, 64], 10)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()
        device = 'cpu'
        file_path = "./models/mnist_model.pth"

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # Example for MNIST
            transforms.Lambda(lambda x: x.view(-1))
        ])

        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
                
        batch_size = 64  # Example batch size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        trainer = MnistTrainer(model, optimizer, loss_fn, train_loader, device=device)
        trainer.train(10)
        trainer.save_model(file_path)


if __name__ == "__main__":
    main()
