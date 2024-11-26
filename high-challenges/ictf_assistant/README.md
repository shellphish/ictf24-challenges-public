# Description

- Name: `ictf_assistant`
- Difficulty / Category: easy / high-school

# Deploy

```bash
docker build -t ictf_assistant -f ./Dockerfile .
docker run -itd -e OPENAI_API_KEY=<OPENAI_KEY> -e ASSISTANT_ID=<ASSISTANT_KEY> --name ictf_assistant_test ictf_assistant
```

This runs the challenge inside a container `ictf_assistant_test` at port 11239. 

Users can visit the challenge through nc:

```bash
nc <IP of the container> 11239
```

# About `OPENAI_API_KEY` and `ASSISTANT_ID`

DO NOT GIVE `OPENAI_API_KEY` or `ASSISTANT_ID` to players. The keys are currently billed under Prof. Wenke Lee's lab account and will be destroyed after iCTF 2024 ends.

`OPENAI_API_KEY` is for accessing openai's models. `ASSISTANT_ID` should be set to the provided value above to avoid re-generating too many assistants during the competition.

The challenge has a PoW machenism to prevent API usage abuse. Pwning the challenge is OK and will not leak the API key, as players will not be able to get a shell or read other information.

# Deliveries to Players

All the files under the [to_players](./to_players/) directory should be delivered to the players.

```bash
zip ./ictf_assistant.zip -r ./to_players
# Now, deliver ictf_assistant.zip to the players.
```

# Solution

Please refer to the [writeup](./writeup/) directory.