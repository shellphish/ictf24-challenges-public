# Base image
FROM ubuntu:20.04

# Set non-interactive mode for APT to prevent prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update system and install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gnupg \
    openjdk-11-jdk \
    ca-certificates \
    && apt-get clean

# Install Node.js (via Nodesource) - Use Node.js 18.x for npm compatibility
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g npm@latest

# Add the Neo4j repository and install Neo4j
RUN wget -O - https://debian.neo4j.com/neotechnology.gpg.key | apt-key add - \
    && echo "deb https://debian.neo4j.com stable 5" > /etc/apt/sources.list.d/neo4j.list \
    && apt-get update \
    && apt-get install -y neo4j


# Copy the application files into the container
WORKDIR /usr/src/app
COPY package.json ./
RUN npm install
COPY . .

# Copy the initialization script for Neo4j
COPY init.cypher /var/lib/neo4j/import/init.cypher

# Expose ports for Neo4j (7474, 7687) and the app (3000)
EXPOSE 5000 7474 7687

# Command to start both Neo4j and the Node.js application
# CMD ["/bin/bash", "-c", "service neo4j start && sleep 10 && cypher-shell -u neo4j -p thestand -f /var/lib/neo4j/import/init.cypher && node app.js"]

# Command to start both Neo4j and the Node.js application
# CMD ["/bin/bash", "-c", "service neo4j start && sleep 10 && cypher-shell -u neo4j -p neo4j 'ALTER CURRENT USER SET PASSWORD FROM \"neo4j\" TO \"securepassword\"' && cypher-shell -u neo4j -p securepassword -f /var/lib/neo4j/import/init.cypher && node app.js"]

CMD ["/bin/bash", "-c", "service neo4j start && sleep 10 && cypher-shell -u neo4j -p neo4j --database=system 'ALTER CURRENT USER SET PASSWORD FROM \"neo4j\" TO \"securepassword\"' && cypher-shell -u neo4j -p securepassword -f /var/lib/neo4j/import/init.cypher && node app.js"]
