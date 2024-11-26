# extract_message.py

import torch
import numpy as np

# Load the mystery model's state dictionary
state_dict = torch.load('mystery_model.pt', weights_only=True)
# state_dict = torch.load('model.pt', weights_only=True)

# Select the same layer used for embedding
layer_name = 'fc1.weight'
weights = state_dict[layer_name]

# import pdbr
# pdbr.set_trace()

# exit()

# Function to extract bits from weights
def extract_bits_from_weights(weights, num_bits):
    flat_weights = weights.view(-1).clone().detach().numpy()
    # Scale weights to integers
    scaled_weights = np.round(flat_weights * (2**8)).astype(np.int32)
    # Extract bits from the least significant bit of each weight
    bits = ''
    for i in range(num_bits):
        bit = scaled_weights[i] & 1
        bits += str(bit)
    return bits

# Calculate the number of bits to extract (length of the message in bits)
message_length = 8 * len("ictf{Neural_Networks_Can_Keep_Secrets!}")
extracted_bits = extract_bits_from_weights(weights, message_length)

# Convert bits back to the message
def bits_to_message(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        chars.append(chr(int(byte, 2)))
    message = ''.join(chars)
    return message

hidden_message = bits_to_message(extracted_bits)
print(f"Hidden message: {hidden_message}")
