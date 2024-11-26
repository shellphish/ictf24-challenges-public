FROM python:3.12.3-slim

# Install dependencies
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --no-install-recommends -y socat \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /app
COPY ./src/* /app/

# Install Python dependencies
WORKDIR /app
RUN chmod +x serve.sh main.py

EXPOSE 25932
CMD ["./serve.sh"]
