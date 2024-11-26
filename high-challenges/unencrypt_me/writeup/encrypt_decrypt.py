import torch
import pickle

def xor_encrypt_decrypt_tensor(tensor: torch.Tensor, key: int) -> torch.Tensor:
    # Convert the tensor to an integer type for XOR, then clone to avoid in-place modifications
    flat_tensor = tensor.view(-1).to(torch.int32).clone()
    
    # Apply XOR encryption/decryption
    for i in range(flat_tensor.numel()):
        flat_tensor[i] = flat_tensor[i] ^ key
    
    # Convert back to the original tensor type and reshape
    return flat_tensor.view(tensor.size()).to(tensor.dtype)

def encrypt(model_state_dict: dict, key: int) -> dict:
    # Apply XOR encryption on each tensor in the state_dict
    encrypted_state_dict = {}
    for name, tensor in model_state_dict.items():
        encrypted_state_dict[name] = xor_encrypt_decrypt_tensor(tensor, key)
    return encrypted_state_dict

def decrypt(model_state_dict_enc: dict, key: int) -> dict:
    # Apply XOR decryption on each tensor in the encrypted state_dict
    decrypted_state_dict = {}
    for name, tensor in model_state_dict_enc.items():
        decrypted_state_dict[name] = xor_encrypt_decrypt_tensor(tensor, key)
    return decrypted_state_dict