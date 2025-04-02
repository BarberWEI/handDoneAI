import torch
import torch.nn as nn

class PigModel(nn.Module):
    def __init__(self):
        super(PigModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64),   # 4 inputs
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  
            nn.Softmax(dim=1)# 2 outputs for the two possible actions, pass or hit
        )

    def forward(self, x):
        return self.network(x)


class MnistModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        Initializes the deep neural network.
        
        Args:
            input_size (int): Size of the input features.
            hidden_sizes (list of int): List with the number of neurons for each hidden layer.
            output_size (int): Size of the output layer.
        """
        super(MnistModel, self).__init__()
        layers = []
        last_size = input_size
        
        # Create hidden layers with ReLU activation
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.ReLU())
            last_size = hidden_size
            
        # Create the output layer
        layers.append(nn.Linear(last_size, output_size))
        
        # Combine layers into a Sequential module
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Defines the forward pass of the network.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.network(x)
