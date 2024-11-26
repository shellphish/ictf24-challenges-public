FROM ubuntu:22.04

RUN apt-get update -y \
    && apt-get install --no-install-recommends -y python3 socat python3-pip gcc build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install requests

#COPY ./src/flag.txt /flag.txt
#RUN chmod 644 /flag.txt

RUN useradd -u 31337 -ms /bin/bash challenge
# Prevents the user from writing to the home directory
RUN chown root:root /home/challenge
RUN chmod 755 /home/challenge

# Source files
COPY src /home/challenge/src
RUN chmod +x /home/challenge/src/run.sh
# Read-only files
#COPY ro /home/challenge/ro
# Writeable files
COPY rw /home/challenge/rw
RUN chown challenge:challenge /home/challenge/rw
RUN chmod 755 /home/challenge/rw

# Creates the application

# Install the service
WORKDIR /home/challenge/src
RUN gcc -g mayor_setter.c -o mayor_setter
RUN chmod 755 /home/challenge/src/mayor_setter

# export llmapi env vars
ENV OPENAI_BASE_URL=http://<LLM_ENDPOINT>
ENV OPENAI_API_KEY=<YOUR_API_KEY>
ENV OPENAI_MODEL=oai-gpt-4o-mini

# The port this leaves on
ENV TERM=xterm
EXPOSE 11301
CMD ["socat", "TCP-LISTEN:11301,reuseaddr,fork", "EXEC:'/home/challenge/src/run.sh',pty,rawer,stderr,su=challenge"]
