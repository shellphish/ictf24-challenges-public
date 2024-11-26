FROM ubuntu:20.04

RUN apt-get update -y \
    && apt-get install --no-install-recommends -y build-essential libmpc-dev libgmp-dev libmpfr-dev python3-dev xinetd python3 python3-pip netcat \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install openai=="1.50.2" gmpy2=="2.1.0" fire

RUN useradd -u 31337 -ms /bin/bash challenge
RUN chown root:root /home/challenge

# Source files
COPY src /home/challenge/src
WORKDIR /home/challenge

RUN chmod 755 -R /home/challenge/
RUN mv /home/challenge/src/xinetd.conf /etc/xinetd.d/aicalc

ENV OPENAI_API_KEY=<OPENAI_API_KEY>
ENV ASSISTANT_ID=<ASSISTANT_ID>
# The port this leaves on
EXPOSE 11239
CMD ["/usr/sbin/xinetd", "-dontfork"]
