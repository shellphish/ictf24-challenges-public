### Exploit 🧨

#### To build
```rename .env.example to .env```

```docker build -t hackchatai .```

```docker run -d -p 5001:5001 hackchatai```

<b>Note</b>: You might need to change model version in `chatbot/chat.py`
from `oai-gpt-4o-mini` to `gpt-4o-mini` if you are using openAI API , here we are proxying through LiteLLM for our deployment.