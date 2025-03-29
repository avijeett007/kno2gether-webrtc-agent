FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    portaudio19-dev \
    ffmpeg \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for LiveKit
EXPOSE 8080

# Environment variables set in docker-compose or by orchestration
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "nhs_agents.py"] 