# Encrypted Model Challenge Walkthrough

## Challenge Overview

A student encrypted a PyTorch model's weights using XOR encryption but forgot the key. The encrypted model is provided in `model_encrypted.pth`. Our goal is to retrieve the key and decrypt the model.

## Understanding the Problem

- **XOR Encryption**: A symmetric encryption method where data is encrypted by performing an XOR operation with a key.
- **PyTorch Model Weights**: Saved as serialized objects, often using Python's `pickle` module.

## Steps to Solve

### 1. Analyze the Encrypted File

Load the encrypted model file and observe its structure.

```python
with open('model_encrypted.pth', 'rb') as f:
    encrypted_data = f.read()
```

### 2. Recognize the File Format
PyTorch model files saved with torch.save() are typically in a specific binary format that includes identifiable headers.

### 3. Known Plaintext Attack
Since we know the structure of a PyTorch model file, we can use a known plaintext attack.

* Assumption: Some parts of the plaintext (original model file) are known or can be guessed.
* Goal: Recover the key by XORing the known plaintext with the ciphertext.

### 4. Obtain Known Plaintext
Identify common headers or patterns in PyTorch model files. All `.pth` files start with `PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0e\x00\x14\x00{<name-of-model-class>}/data.pkl`

### 5. Calculate the Key
Truncate the known plaintext to match the length of the encrypted data if necessary.

```python
key_length = 16

key = bytearray(key_length)
for i in range(len(known_plaintext)):
    key[i % key_length] = encrypted_data[i] ^ known_plaintext[i]

full_key = (key * (len(encrypted_data) // key_length + 1))[:len(encrypted_data)]
```

### 6. Decrypt the Encrypted Data
Now that we have the key, decrypt the entire encrypted file.

```python
decrypted_data = bytes([c ^ k for c, k in zip(encrypted_data, full_key)])
```

### 7. Load the Decrypted Model
Write the decrypted data to a file and load the model.

```python
with open('model_decrypted.pth', 'wb') as f:
    f.write(decrypted_data)

model = torch.load('model_decrypted.pth')
```

### 8. Retrieve the Flag
The flag is stored within the model's state or attributes.

```python
print(model.get('flag'))
```