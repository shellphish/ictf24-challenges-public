import torch
from challenge.model import Model
import cv2
import torch.nn as nn
def invert_linear(linear_layer, y):
    """
    Inverts a linear layer transformation
    y = Wx + b -> x = W⁻¹(y - b)
    """
    W = linear_layer.weight.data
    if linear_layer.bias is not None:
        y = y - linear_layer.bias.data
    
    # Since y = Wx, we need x = W^(-1)y
    # Using transpose since for many networks W^T ≈ W^(-1)
    x = torch.mm(y, W)  # Using regular matrix multiplication with transposed weight
    return x

def recover_image(garbled_image, model):
    """
    Recovers the original image by inverting the transformations of two linear layers.
    """
    # Flatten image if needed
    if len(garbled_image.shape) > 2:
        garbled_image = garbled_image.reshape(garbled_image.shape[0], -1)
        
    # Recover x from garbled_image by going backwards through the layers
    intermediate = invert_linear(model.linear2, garbled_image)
    original = invert_linear(model.linear1, intermediate)
    
    # Reshape back to original image dimensions if needed
    if len(garbled_image.shape) > 2:
        original = original.reshape(garbled_image.shape)
        
    return original

# Load model
model = Model()
model.load_state_dict(torch.load('challenge/model.pth'))

# Load and process the image
flag = cv2.imread('challenge/out.png', 0)
tensor = torch.from_numpy(flag).float()

# Recover the original image
recovered_image = recover_image(tensor, model)

# Convert to uint8 and save the image
cv2.imwrite('solve_out.png', recovered_image.detach().numpy().astype('uint8'))