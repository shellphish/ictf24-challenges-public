FROM ubuntu:22.04

# Install dependencies
RUN apt-get update -y \
    && apt-get install --no-install-recommends -y coreutils build-essential socat python3 python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install requests openai rich

# Create a new user (su=challenge in socat)
RUN useradd -u 31337 -ms /bin/bash challenge
RUN useradd -u 31338 -M -s /bin/bash gpt
RUN passwd -d gpt
# Prevents the user from writing to the home directory
RUN chown root:root /home/challenge
RUN chmod 755 /home/challenge
RUN mkdir /gpt && chmod 755 /gpt

# Copy files
# COPY ./src/flag.txt /flag.txt
COPY src /home/challenge/src

# Set permissions
# RUN chmod 644 /flag.txt
RUN cp /home/challenge/src/flag.txt /gpt/flag.txt && chmod 644 /gpt/flag.txt
RUN chmod 755 /home/challenge/src/challenge.py
RUN chmod 755 /home/challenge/src/service.sh

# Install the service
WORKDIR /home/challenge/src

# export llmapi env vars
ENV OPENAI_BASE_URL=http://<LLM_ENDPOINT>
ENV OPENAI_API_KEY=<YOUR_API_KEY>
ENV OPENAI_MODEL=oai-gpt-4o-mini

# The port this leaves on
ENV TERM=xterm
EXPOSE 11301
CMD ["socat", "TCP-LISTEN:11301,reuseaddr,fork", "EXEC:'/home/challenge/src/service.sh',pty,rawer,stderr,su=challenge"]

# docker rm -f $(docker ps -a -q --filter ancestor=mychallenge); docker build -t mychallenge . && docker run -d -p 11301:11301 mychallenge
# nc localhost 11301