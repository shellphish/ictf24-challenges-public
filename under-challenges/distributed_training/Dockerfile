FROM ubuntu:22.04

RUN apt-get update -y \
    && apt-get install --no-install-recommends -y python3 python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install flask flask-limiter numpy
COPY ./src/flag.txt /flag.txt
RUN chmod 644 /flag.txt

RUN useradd -u 31337 -ms /bin/bash challenge
RUN useradd -u 31338 -M -s /bin/bash compute
RUN passwd -d compute
# Prevents the user from writing to the home directory
RUN chown root:root /home/challenge
RUN chmod 755 /home/challenge
RUN mkdir /work && chmod 755 /work

# Source files
COPY src /home/challenge/src
RUN cp /home/challenge/src/flag.txt /work/flag.txt && chmod 644 /work/flag.txt

# Creates the application
WORKDIR /home/challenge/src

# The port this leaves on
EXPOSE 32123
USER challenge
CMD ["python3", "app.py"]