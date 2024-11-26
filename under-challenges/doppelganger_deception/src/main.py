import argparse
import sys
import torch
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch
import numpy as np
import random

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(f'Using device: {device}')

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # Fully connected layers
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.dropout2d(x, 0.25, training=self.training)
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 320)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.dropout(x, 0.5, training=self.training)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)


def load_MNIST(batch_size):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('.', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                    ])),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader


def train(model_filename):
    # hyperparameters
    batch_size = 64
    epochs = 10  # For brevity; increase for better accuracy if desired

    # Initialize the network and optimizer
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train_loader = load_MNIST(batch_size=batch_size)

    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device) 
            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()

    # Save the trained model
    torch.save(model.state_dict(), model_filename)


def test(model_filename):
    state_dict = torch.load(model_filename, weights_only=True)

    model = Net().to(device)
    model.load_state_dict(state_dict)

    # Test the model on the MNIST test set
    test_loader = load_MNIST(batch_size=1000)

    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    print(f"Accuracy: {correct / len(test_loader.dataset):.2%}")


def main():
    parser = argparse.ArgumentParser(description='Train or test the mystery model')
    parser.add_argument('mode', choices=['train', 'test'], help='train or test the mystery model')
    parser.add_argument('model_filename', help='filename to save/load the model')
    args = parser.parse_args()

    if args.mode == 'train':
        train(args.model_filename)
    elif args.mode == 'test':
        test(args.model_filename)
    else:
        print("Invalid mode")
        sys.exit(1)


if __name__ == '__main__':
    main()