FROM ubuntu:22.04

RUN apt-get update -y \
    && apt-get install --no-install-recommends -y xinetd  python3 python3-pip netcat \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install langchain-community chromadb

RUN useradd -u 31337 -ms /bin/bash challenge

# Prevents the user from writing to the home directory
RUN chown root:root /home/challenge
RUN chmod 755 /home/challenge

# Set up cache environment variable
ENV XDG_CACHE_HOME=/home/challenge/.cache
RUN mkdir -p /home/challenge/.cache && chown -R challenge:challenge /home/challenge/.cache

# Source files
COPY src /home/challenge/src
COPY src/flag /home/challenge/flag
COPY src/flag /flag

# Creates the application
WORKDIR /home/challenge

# make the run script executable
RUN chmod +x /home/challenge/src/run_chall.sh

# Install the service
RUN mv /home/challenge/src/xinetd.conf /etc/xinetd.d/chall

USER challenge
RUN echo "2" | /home/challenge/src/run_chall.sh

# The port this leaves on
EXPOSE 7462
CMD ["/usr/sbin/xinetd", "-dontfork"]
