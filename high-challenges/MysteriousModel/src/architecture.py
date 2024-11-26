import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Define device (use GPU if available)
device = 'cpu'
NUM_CLASSES = 6

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES) # THE OUTPUT LOGITS, EACH CORRESPOND TO A DIGIT
        # to learn what a logit is, see: https://datascience.stackexchange.com/questions/31041/what-does-logits-in-machine-learning-mean
        # HINT: to get the predicted probabilities from output logits, use the softmax function
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to load the model
def load_model(filepath='mysterious_model.pth', device='cpu'):
    model = SimpleCNN().to(device)
    checkpoint = torch.load(filepath, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Model loaded from {filepath}')
    return model.eval()

# display an individual image in the test data
def display_image(image:np.ndarray):
    img = (image*255).astype(np.uint8)
    plt.figure()
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)