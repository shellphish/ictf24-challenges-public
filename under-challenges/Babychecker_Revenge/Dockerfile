# Use the official Python image as a base
FROM python:3.9

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000
ENV FLAG=ictf{U_4r3_55RF_Exp3e7_N0w}

# export llmapi env vars
ENV OPENAI_BASE_URL=http://<LLM_ENDPOINT>
ENV OPENAI_API_KEY=<YOUR_API_KEY>
ENV OPENAI_MODEL=oai-gpt-4o-mini

# Set the working directory
WORKDIR /app

# Copy the requirements file if you have one (optional)
# If you don't have a requirements.txt, you can skip this step
# COPY requirements.txt .

# Install necessary packages
RUN pip install --no-cache-dir Flask openai requests

# Copy the application code to the container
COPY ./src/ /app/ 

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["flask", "run"]
