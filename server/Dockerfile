# Description: Dockerfile for the server
FROM python:3.12

# Copy the code
COPY src /app
# Copy the keyfile
COPY gcp-project-api-credentials.json /app/keyfile.json
# Set the working directory
WORKDIR /app
# Add everything to the working directory
ADD . /app

# Set the Environment variables
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/keyfile.json

# Install dependencies
RUN pip install --no-cache-dir -r src/requirements.txt

# Network port
EXPOSE 8080

# Run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
