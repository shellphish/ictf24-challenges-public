#!/usr/bin/env python3

"""
### TO USE ANY OPENAI MODEL:
* Update the value of OPENAI_API_KEY below (sk-YOUR-API-KEY) with your OpenAI API key
* Then choose a model (e.g., `gpt-4o-mini`) and use the OpenAI API as you normally would


---
### IF YOU NEED TO USE ANY OTHER MODEL:
1) Please let us know in advance so that we can prepare the competition's LiteLLM server
2) Refer to https://github.com/BerriAI/litellm to setup a LiteLLM server
3) Update the value of OPENAI_BASE_URL (below) with the LiteLLM url
4) Update the value of OPENAI_API_KEY (below) with your non-openai API key
"""

import requests

if __name__ == "__main__":
    # token = input('Enter your token: ')
    with open('/tmp/ictf.env', 'w') as f:
        f.write('export OPENAI_BASE_URL=http://<LLM_ENDPOINT>\n')
        f.write('export OPENAI_API_KEY=<OPENAI_KEY>\n')
        f.write('export OPENAI_MODEL=oai-gpt-4o-mini\n')
        
