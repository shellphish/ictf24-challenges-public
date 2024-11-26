# First stage - use Selenium image
FROM selenium/standalone-chrome:latest as selenium

# Second stage - fresh start
FROM scratch

# Copy everything from the selenium stage
COPY --from=selenium / /

# Run as root
USER root

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    sqlite3

# Copy the application code
COPY . /app/

# Create and activate a virtual environment, then install dependencies
RUN python3 -m venv venv
RUN . venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

# Only expose the port we want
EXPOSE 5001

ENV OPENAI_BASE_URL=http://<LLM_ENDPOINT>
ENV OPENAI_API_KEY=<YOUR_API_KEY>
ENV OPENAI_MODEL=oai-gpt-4o-mini
ENV CTF_FLAG=ictf{well_d0n3_on_s0lv1ng_th1s_ch4ll3ng3_hackchatai24}

# Set the entry point to run the Flask app
CMD ["sh", "-c", ". venv/bin/activate && python3 app.py"]
