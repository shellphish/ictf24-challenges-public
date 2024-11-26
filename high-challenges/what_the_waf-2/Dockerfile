FROM ubuntu:22.04 as base

# Set noninteractive installation to skip prompts
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y coreutils build-essential libsasl2-dev python-dev-is-python3 python3 python3-pip libldap2-dev libssl-dev slapd ldap-utils vim sudo
RUN pip install --upgrade pip
RUN pip install flask-simpleldap langchain langchain_openai Flask-Limiter rich openai requests

# Pre-seed the configuration for slapd using /var/lib/dpkg/info/slapd.templates
RUN echo "slapd slapd/password1 password megalol" | debconf-set-selections && \
    echo "slapd slapd/password2 password megalol" | debconf-set-selections && \
    echo "slapd slapd/no_configuration boolean false" | debconf-set-selections && \
    echo "slapd slapd/domain string example.com" | debconf-set-selections && \
    echo "slapd shared/organization string 'example.com'" | debconf-set-selections

# Copy the users!
COPY src/ldap-data/init.ldif /tmp/init.ldif

# Reconfigure slapd to apply the settings
RUN dpkg-reconfigure -f noninteractive slapd

# Start the sladp and add users
RUN service slapd start && ldapadd -x -D "cn=admin,dc=example,dc=com" -w megalol -f /tmp/init.ldif


COPY src/webapp /webapp
COPY src/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# export llmapi env vars
ENV OPENAI_BASE_URL=http://<LLM_ENDPOINT>
ENV OPENAI_API_KEY=<YOUR_API_KEY>
ENV OPENAI_MODEL=oai-gpt-3.5-turbo

EXPOSE 9999

ENTRYPOINT ["/entrypoint.sh"]
