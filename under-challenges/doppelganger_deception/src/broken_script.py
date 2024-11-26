# Convert the message to binary
    bits = ''.join(format(ord(c), '08b') for c in message)

# Embed the message bits into the least significant bits of the weights
    # Scale weights to integers
    scaled_weights = np.round(flat_weights * (2**8)).astype(np.int32)
    # Embed bits into the least significant bit of each weight
        scaled_weights[i] = (scaled_weights[i] & ~1) | bit  # Set LSB to the bit
    # Convert back to original scale
    modified_weights = scaled_weights.astype(np.float32) / (2**8)
    # Reshape to original shape

# Select a layer to embed the message (e.g., the first fully connected layer)

# Embed the message into the weights
# Update the state dictionary with modified weights
state_dict[layer_name] = modified_weights
# Save the modified state dictionary as the mystery model
