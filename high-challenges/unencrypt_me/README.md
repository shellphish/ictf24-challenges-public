# Unencrypt Me

A student encrypted their trained PyTorch model weights but forgot the encryption key. All he remembers is that he used an 8-bit encryption key. You have access to the test dataset on which the model performed exceptionally well. Your task is to help the student decrypt the model.

## Getting Started

- **Files Provided**:
  - `encrypted_regression_model.pth`: the encrypted weights of the model
  - `encrypt_decrypt.py`: the encryption/decryption scripts
  - `test_data.pt`: the test data
  - `evaluate.py`: the evaluation scripts
- **Objective**: Find the encryption key that correctly decrypts the model weights.
- **Flag Format**: `ictf{<correct_encryption_key>}`

Good luck!
