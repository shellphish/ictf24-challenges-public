import torch
import torch.nn as nn
from encrypt_decrypt import decrypt

# Define the same model architecture
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.hidden = nn.Linear(1, 10)  # Hidden layer with 10 neurons
        self.relu = nn.ReLU()           # Non-linear activation function
        self.output = nn.Linear(10, 1)  # Output layer

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

min = 1000000000000
true_key = 0
for key in range(256):
    # Initialize the model
    model = RegressionModel()

    # Load the weights into the model
    state_dict = torch.load("encrypted_regression_model.pth")
    state_dict = decrypt(state_dict, key)
    model.load_state_dict(state_dict)
    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully.")

    # Load the test data
    test_data = torch.load("test_data.pt")
    X_test = test_data['X']
    y_test = test_data['y']

    # Evaluate the model
    predictions = model(X_test)
    loss = nn.MSELoss()(predictions, y_test)
    if loss < min:
        min = loss
        true_key = key
    print(f"Mean squared error: {loss.item()}")

print(f"True key: {true_key}")