import torch

# Read the encrypted data
with open('../src/model_encrypted.pth', 'rb') as f:
    encrypted_data = f.read()

# Known plaintext: the first few bytes of a pickle file saved by torch.save()
# This is the first part of the header and the path to the data file
known_plaintext = b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00'

key_length = 16

# Recover the key using the known plaintext
key = bytearray(key_length)
for i in range(len(known_plaintext)):
    key[i % key_length] = encrypted_data[i] ^ known_plaintext[i]

# Now, we can extend the key to the length of the encrypted data
full_key = (key * (len(encrypted_data) // key_length + 1))[:len(encrypted_data)]

# Decrypt the data
decrypted_data = bytes([c ^ k for c, k in zip(encrypted_data, full_key)])

# Write the decrypted data to a file
with open('model_decrypted.pth', 'wb') as f:
    f.write(decrypted_data)

# Try to load the decrypted model
try:
    model = torch.load('model_decrypted.pth')
    # Retrieve and print the flag
    print(model.get('flag'))
except Exception as e:
    print("Failed to load the model:", e)
