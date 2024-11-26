# MATERIALS 
This should be enough for you to find your way through this challenge.
## hash cracking 
1. hashing, a quick primer: https://learning.quantum.ibm.com/coursepractical-introduction-to-quantum-safe-cryptography/cryptographic-hash-functions
2. attacks on hash functions: some functions are just broken. For example, look into MD5 in the above link. 
3. tools: have you ever heard of dcode? https://www.dcode.fr/en

## secrets in your code 
1. you should never hardcode sensitive information in your code. What does "sensitive" mean? Hashes are not ciphertext, they might be misused to conceal secrets and embedded into text files. That is bad practice, and can lead to sensitive information leaks. 