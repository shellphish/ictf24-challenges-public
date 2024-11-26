FROM ubuntu:20.04

# Install dependencies
RUN apt-get update -y \
    && apt-get install --no-install-recommends -y coreutils build-essential socat python3 python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install requests openai rich

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
RUN chmod 755 /home/challenge/src/rate_limiter.py
RUN chmod 755 /home/challenge/src/challenge.py
RUN chmod 755 /home/challenge/src/service.sh

# Install the service
WORKDIR /home/challenge/src
ENV TERM=xterm

# The port this leaves on
EXPOSE 12567
CMD ["socat", "TCP-LISTEN:12567,reuseaddr,fork", "EXEC:'/home/challenge/src/service.sh',pty,rawer,stderr,su=challenge"]

# docker rm -f $(docker ps -a -q --filter ancestor=mychallenge); docker build -t mychallenge . && docker run -d -p 11301:11301 mychallenge
# nc localhost 11301