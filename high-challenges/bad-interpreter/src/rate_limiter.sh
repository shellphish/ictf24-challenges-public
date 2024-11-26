#!/bin/bash

echo "Enter your token: "
read -t 30 token

export OPENAI_BASE_URL=<LITELLM_ENDPOINT>
export OPENAI_API_KEY=$token
export OPENAI_MODEL=oai-gpt-4o-mini
