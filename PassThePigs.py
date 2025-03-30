import random

class PassThePigs:
    # Static constant arrays
    PIG_VALUE = [15, 10, 5, 5, 0, 0]
    PIG_NAMES = ["Leaning Jowler", "Snouter", "Trotter", "Razorback", "No Dot", "Dot"]

    def __init__(self, amount_of_players, print_game=True):
        self.players_bank = [0] * amount_of_players
        self.print_game = print_game
        self.hand_value = 0

    def get_hand_value(self):
        return self.hand_value

    # Returns the value of the pigs based on the pig types rolled
    def value_of_pigs(self):
        pig_status = self.get_pigs_status()
        pig1, pig2 = pig_status[0], pig_status[1]
        if pig1 + pig2 == 9:
            return 0
        elif pig1 == pig2:
            if pig1 == 0:
                return 60
            elif pig1 == 1:
                return 40
            elif pig1 == 2 or pig1 == 3:
                return 20
            else:
                return 1
        else:
            return self.PIG_VALUE[pig1] + self.PIG_VALUE[pig2]

    # Gets the status (i.e. type) of the two pigs rolled
    def get_pigs_status(self):
        pigs_status = [self.get_pig_role(), self.get_pig_role()]
        if self.print_game:
            print(f"{self.PIG_NAMES[pigs_status[0]]} and a {self.PIG_NAMES[pigs_status[1]]}", end="")
        return pigs_status

    # Determines the role (or type) of a pig based on a random number
    def get_pig_role(self):
        random_number = random.random()
        if random_number <= 0.007:
            return 0
        elif random_number <= 0.037:
            return 1
        elif random_number <= 0.125:
            return 2
        elif random_number <= 0.349:
            return 3
        elif random_number <= 0.651:
            return 4
        else:
            return 5

    # Returns the bank values of all players except the one at player_number
    def get_players_bank_values(self, player_number):
        return [bank for i, bank in enumerate(self.players_bank) if i != player_number]

    # Simulate a player's pig roll turn and update hand_value accordingly
    def player_role_pigs(self, player_number):
        value = self.value_of_pigs()
        if value != 0:
            self.hand_value += value
            if self.print_game:
                print(f" for a roll of {value}. hand score is now {self.hand_value}.")
            return False
        else:
            self.hand_value = 0
            if self.print_game:
                print(f" for a roll of {value} hand score is now {self.hand_value} that's a Pig Out!")
            return True

    # After a round, add the hand value to the player's bank and reset hand_value
    def change_player_bank_after_round(self, player_number):
        self.players_bank[player_number] += self.hand_value
        self.hand_value = 0

    # Sets a player's bank value directly
    def set_player_bank(self, player_number, value):
        self.players_bank[player_number] = value

    # Retrieves a player's bank value
    def get_player_bank(self, player_number):
        return self.players_bank[player_number]
