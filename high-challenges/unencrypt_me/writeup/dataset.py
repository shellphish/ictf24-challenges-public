import torch

# Generate synthetic data
torch.manual_seed(42)  # For reproducibility
X_train = torch.linspace(-1, 1, 100).reshape(-1, 1)
y_train = 3 * X_train**2 + 2 * X_train + 1 + 0.1 * torch.randn(X_train.size())  # Adding some noise

X_test = torch.linspace(-1, 1, 20).reshape(-1, 1)
y_test = 3 * X_test**2 + 2 * X_test + 1 + 0.1 * torch.randn(X_test.size())  # Adding some noise

# Save the data as a dictionary in a .pt file
train_data = {
    'X': X_train,
    'y': y_train
}
test_data = {
    'X': X_test,
    'y': y_test
}

torch.save(train_data, "train_data.pt")
torch.save(test_data, "test_data.pt")
print("Data saved as 'train_data.pt' and 'test_data.pt'")