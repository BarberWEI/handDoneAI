import torch
import torch.nn as nn

class Predictor:
    def __init__(self, model_class, model_path, input_size, hidden_sizes, output_size, device='cpu'):
        """
        Initializes the Predictor by loading a saved model.
        
        Args:
            model_class (nn.Module): The model class (not an instance).
            model_path (str): Path to the saved model's state_dict.
            input_size (int): Input feature size (e.g., 784 for MNIST).
            hidden_sizes (list of int): Hidden layer sizes.
            output_size (int): Number of output classes.
            device (str): 'cpu' or 'cuda'.
        """
        self.device = torch.device(device)
        
        # Instantiate the model architecture
        self.model = model_class(input_size, hidden_sizes, output_size).to(self.device)
        
        # Load saved weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()  # Set to eval mode

    def predict(self, input_tensor):
        """
        Predicts the class of a given input tensor.
        
        Args:
            input_tensor (torch.Tensor): A single input tensor (already flattened and normalized).
        
        Returns:
            int: Predicted class label.
        """
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            output = self.model(input_tensor)
            predicted_class = output.argmax(dim=1).item()
            return predicted_class
