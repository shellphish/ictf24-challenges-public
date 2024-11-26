import torch
import random

# Create a simple model or data to save
model = {
    'state_dict': {'weight': torch.randn(10, 10)},
    'flag': 'ictf{XOR_Encryption_Is_Not_Secure}'
}

# Save the model
torch.save(model, 'model.pth')

# Read the saved model file
with open('model.pth', 'rb') as f:
    original_data = f.read()

# Generate a random key
key_length = 16  # Example key length
key = bytes(random.getrandbits(8) for _ in range(key_length))

# Encrypt the data using XOR
encrypted_data = bytes([b ^ key[i % len(key)] for i, b in enumerate(original_data)])

# Write the encrypted data to a new file
with open('model_encrypted.pth', 'wb') as f:
    f.write(encrypted_data)
