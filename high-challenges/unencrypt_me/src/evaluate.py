import torch
import torch.nn as nn

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

# Initialize the model
model = RegressionModel()

# Load the weights into the model
state_dict = torch.load("encrypted_regression_model.pth")
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
print(f"Mean squared error: {loss.item()}")
