FROM python:3.11-slim

# Required label for Hugging Face Spaces + OpenEnv discovery
LABEL org.opencontainers.image.title="email-triage-env"
LABEL org.opencontainers.image.description="Email Triage & Response Environment for OpenEnv"
LABEL hf_space="openenv"

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Expose env server port (HF Spaces requires 7860)
EXPOSE 7860

# Default: run the environment server
CMD ["python", "server.py"]
