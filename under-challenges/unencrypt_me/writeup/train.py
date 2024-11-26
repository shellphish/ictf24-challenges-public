import torch
import torch.nn as nn
from tqdm import tqdm
from encrypt_decrypt import encrypt

# Define a more complex neural network model
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

# Instantiate and train the model
model = RegressionModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Load training and test data
train_data = torch.load("train_data.pt")
test_data = torch.load("test_data.pt")
X_train = train_data['X']
y_train = train_data['y']
X_test = test_data['X']
y_test = test_data['y']

# Train the model (shortened for saving purpose)
num_epochs = 500
for epoch in tqdm(range(num_epochs)):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    loss.backward()
    optimizer.step()

# Save model state
torch.save(model.state_dict(), "regression_model.pth")

# Encrypt the model state
encrypted_model = encrypt(model.state_dict(), 189)
torch.save(encrypted_model, "encrypted_regression_model.pth")
