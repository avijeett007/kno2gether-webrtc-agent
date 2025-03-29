# Build stage
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libportaudio2 \
    portaudio19-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
COPY requirements.txt .

# Install PyTorch first separately (better stability)
RUN pip install --no-cache-dir torch==2.0.0

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libportaudio2 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only necessary files
COPY nhs_agents.py .

# Create necessary directories
RUN mkdir -p ./data
RUN mkdir -p ./model_cache

# Create empty .env file (will be populated via environment variables or volume mount)
RUN touch .env

# Set environment variable for model cache directory
ENV LIVEKIT_AGENTS_CACHE_DIR=/app/model_cache
ENV PYTHONUNBUFFERED=1

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD ps aux | grep python | grep nhs_agents.py || exit 1

# Create startup script that will download models then start the agent
RUN echo '#!/bin/bash\n\
echo "Downloading required models..."\n\
python nhs_agents.py download-files\n\
echo "Starting NHS Virtual Assistant..."\n\
exec python nhs_agents.py start' > /app/start.sh && \
    chmod +x /app/start.sh

# Run the startup script
CMD ["/app/start.sh"] 