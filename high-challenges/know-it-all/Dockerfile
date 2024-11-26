FROM ubuntu:20.04

RUN apt-get update -y \
    && apt-get install --no-install-recommends -y xinetd python3 python3-pip netcat\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install openai
RUN pip install requests

RUN useradd -u 31337 -ms /bin/bash challenge
# Prevents the user from writing to the home directory
RUN chown root:root /home/challenge
RUN chmod 755 /home/challenge

# Source files
COPY src /home/challenge/src

# Creates the application
WORKDIR /home/challenge

# Install the service
RUN mv /home/challenge/src/xinetd.conf /etc/xinetd.d/sampleremote

# The port this leaves on
EXPOSE 7462

ENV OPENAI_BASE_URL=http://<LLM_ENDPOINT>
ENV OPENAI_API_KEY=<YOUR_API_KEY>
ENV OPENAI_MODEL=oai-gpt-4o-mini

CMD ["/usr/sbin/xinetd", "-dontfork"]
