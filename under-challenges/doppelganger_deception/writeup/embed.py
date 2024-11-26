# embed_message.py

import torch
import numpy as np

# Load the trained model's state dictionary
state_dict = torch.load('model.pt', weights_only=True)

# Define the secret message
secret_message = "ictf{Neural_Networks_Can_Keep_Secrets!}"

# Convert the message to binary
def message_to_bits(message):
    bits = ''.join(format(ord(c), '08b') for c in message)
    return bits

message_bits = message_to_bits(secret_message)
print(f"Message length: {len(message_bits)} bits")

# Embed the message bits into the least significant bits of the weights
def embed_bits_in_weights(weights, bits):
    flat_weights = weights.view(-1).clone().detach().numpy()
    num_bits = len(bits)
    if num_bits > flat_weights.size:
        raise ValueError("Message is too long to embed in the selected weights.")
    # Scale weights to integers
    scaled_weights = np.round(flat_weights * (2**8)).astype(np.int32)
    # Embed bits into the least significant bit of each weight
    for i in range(num_bits):
        bit = int(bits[i])
        scaled_weights[i] = (scaled_weights[i] & ~1) | bit  # Set LSB to the bit
    # Convert back to original scale
    modified_weights = scaled_weights.astype(np.float32) / (2**8)
    # Reshape to original shape
    modified_weights = torch.from_numpy(modified_weights.reshape(weights.shape))
    return modified_weights

# Select a layer to embed the message (e.g., the first fully connected layer)
layer_name = 'fc1.weight'
weights = state_dict[layer_name]

# Embed the message into the weights
modified_weights = embed_bits_in_weights(weights, message_bits)

# Update the state dictionary with modified weights
state_dict[layer_name] = modified_weights

# Save the modified state dictionary as the mystery model
torch.save(state_dict, 'mystery_model.pt')
