import requests
import sys

REMOTE_SERVER = sys.argv[1]
print(f'Connecting to {REMOTE_SERVER}')

if REMOTE_SERVER.startswith('http://'):
    REMOTE_SERVER = REMOTE_SERVER[7:]

# Get / endpoint
response = requests.get(f'http://{REMOTE_SERVER}/')
print(f'GET /: {response.status_code} {response.text}')

# Upload a file
with open('exp_mat.pt', 'rb') as f:
    response = requests.post(f'http://{REMOTE_SERVER}/upload', files={'file': f})
    print(f'POST /upload: {response.status_code} {response.text}')
