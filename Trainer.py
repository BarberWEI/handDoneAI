import torch

class MnistTrainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, device='cuda'):
        """
        Initializes the Trainer.
        
        Args:
            model (nn.Module): The PyTorch model to train.
            optimizer (torch.optim.Optimizer): The optimizer for model parameters.
            loss_fn (callable): The loss function.
            train_loader (DataLoader): PyTorch DataLoader for the training dataset.
            device (str): Device to run the training on ('cpu' or 'cuda').
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.device = device

    def train_epoch(self):
        """
        Trains the model for one epoch.
        
        Returns:
            float: The average loss for the epoch.
        """
        self.model.train()  # Set the model to training mode
        total_loss = 0.0

        for batch in self.train_loader:
            # Assume each batch is a tuple of (inputs, targets)
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()       # Clear gradients
            outputs = self.model(inputs)       # Forward pass
            loss = self.loss_fn(outputs, targets)  # Compute loss
            loss.backward()                  # Backpropagation
            self.optimizer.step()            # Update parameters

            total_loss += loss.item()

        average_loss = total_loss / len(self.train_loader)
        return average_loss

    def train(self, num_epochs, print_every=1):
        """
        Runs the training loop for a given number of epochs.
        
        Args:
            num_epochs (int): Number of epochs to train.
            print_every (int): Frequency (in epochs) to print the loss.
        """
        for epoch in range(num_epochs):
            epoch_loss = self.train_epoch()
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")
                
    def save_model(self, file_path):
        """
        Saves the trained model's state dictionary to a file.
        
        Args:
            file_path (str): The path where the model state will be saved.
        """
        torch.save(self.model.state_dict(), file_path)

