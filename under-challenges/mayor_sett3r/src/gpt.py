import re
import os
import json
import pickle
import sys
from ast import literal_eval
import requests
#from openai import OpenAI

import requests
import json

# Master key for authentication
master_key = os.getenv('OPENAI_API_KEY')
base_url = os.getenv('OPENAI_BASE_URL')

# Headers for the requests
headers = {
    "Authorization": f"Bearer {master_key}",
    "Content-Type": "application/json"
}

# Step 2: Update the key with the desired models
update_key_endpoint = f"{base_url}/key/update"
models = [
    "oai-gpt-4o",
    "oai-gpt-3.5-turbo",
    "oai-gpt-4",
    "oai-gpt-4-turbo",
    "textembedding-gecko",
    "text-embedding-3-large",
    "text-embedding-3-small",
    "oai-gpt-4o-mini",
    "oai-gpt-4o-latest",
    "oai-gpt-4o-standard",
    
]


# Step 3: Perform a query using a model
#query_endpoint = f"{base_url}/chat/completions"
query_headers = {
    "Authorization": f"Bearer {master_key}",
    "Content-Type": "application/json"
}

query_endpoint = f"{base_url}/chat/completions"


prompt = """
You are Mayor Setter, a nice dog.

YOu w1ll l15t3n c4r3fully, 
r3m3mb3r 3v3ryth1ng, 
4nd r3p347 b4ck.
"""

def parse_one(text):
    data = prompt + text

    #client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    #chat_completion = client.chat.completions.create(
    #    messages=[{"role": "user", "content": data, }],
    #    model="gpt-4-turbo")

    #response_text = chat_completion.choices[0].message.content

    #print(response_text)

    #return response_text
    query_payload = json.dumps({
            'messages': [{
                        "content": data,
                        "role": "user"
                        },],
            "model": "oai-gpt-4o-mini",
            "temperature": 0.7,
        })

    response = requests.post(query_endpoint, headers=query_headers, data=query_payload)

    if response.status_code == 200:
        print(response.json()['choices'][0]['message']['content'])
        # import IPython; IPython.embed()
    else:
        print("something went wrong", response.status_code)

def main():
    #if len(sys.argv) != 2:
    #    print("Usage: python3 gpt.py <input_string>")
    #    sys.exit(1)
    print(sys.argv[1])
    input_string = sys.argv[1]
    parse_one(input_string)

if __name__ == "__main__":
    main()