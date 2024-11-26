import hashlib 
secret_input = "pizzanduja"
# flag: ictf{calabriornia} 

plain_first = "calabri" 
plain_second = "fornia"

pizzash = hashlib.md5(secret_input[0:5].encode()).hexdigest() 
hashed_first = hashlib.md5(plain_first.encode()).hexdigest()
print(f"first part: {plain_first}, hash: {hashed_first}")
print() 

FIRST = int(pizzash, 16) ^ int(hashed_first, 16)
print("FIRST:\t", format(FIRST, 'x') )


pastash = hashlib.md5(secret_input[5:].encode()).hexdigest()
hashed_second = hashlib.md5(plain_second.encode()).hexdigest()
print(f"second part: {plain_second}, hash: {hashed_second}")
print() 
SECOND = int(pastash, 16)  ^ int(hashed_second, 16)
print("SECOND:\t", format(SECOND, 'x') )

import hashlib
import base64
from xor_cipher import cyclic_xor

def super_secure_encryption(S: bytes) -> bytes:
    return int.from_bytes(cyclic_xor(base64.b64encode(str(S).encode()),b'abcd'))

for i in range(len(secret_input)): 
    print(secret_input[i], "--->", super_secure_encryption(secret_input[i]))
    
    
