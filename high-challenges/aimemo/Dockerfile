FROM ubuntu:20.04

RUN apt-get update -y \
    && apt-get install --no-install-recommends -y coreutils build-essential socat python3 python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install requests openai rich tiktoken

# Create a new user (su=challenge in socat)
RUN useradd -u 31337 -ms /bin/bash challenge
# Prevents the user from writing to the home directory
RUN chown root:root /home/challenge
RUN chmod 755 /home/challenge

# Copy files
# COPY ./src/flag.txt /flag.txt
COPY src /home/challenge/src

# Set permissions
# RUN chmod 644 /flag.txt
RUN chmod 755 /home/challenge/src/challenge.py
RUN chmod 755 /home/challenge/src/service.sh

# Install the service
WORKDIR /home/challenge/src
ENV TERM=xterm

# export llmapi env vars
ENV OPENAI_BASE_URL=http://<LLM_ENDPOINT>
ENV OPENAI_API_KEY=<YOUR_API_KEY>
ENV OPENAI_MODEL=oai-gpt-4o-mini

# The port this leaves on
EXPOSE 11301
CMD ["socat", "TCP-LISTEN:11301,reuseaddr,fork", "EXEC:'/home/challenge/src/service.sh',pty,rawer,stderr,su=challenge"]

# docker rm -f $(docker ps -a -q --filter ancestor=aimemo); docker build -t aimemo . && docker run -d -p 11301:11301 aimemo
# nc localhost 11301