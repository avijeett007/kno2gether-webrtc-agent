# Build stage
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libportaudio2 \
    portaudio19-dev \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements.txt first
COPY requirements-docker.txt ./requirements.txt

# Install PyTorch first separately (better stability)
RUN pip install --no-cache-dir torch==2.0.0

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the agent file into the image
COPY nhs_agents.py .

# Set up directories for models
RUN mkdir -p /root/.cache/livekit-plugins-turn-detector/
RUN mkdir -p /root/.cache/livekit-plugins-silero/

# Run a script to manually download the turn detector model files at build time
RUN echo 'import sys; sys.path.append("/opt/venv/lib/python3.10/site-packages"); from livekit.plugins import turn_detector; model = turn_detector.EOUModel(); print("Turn detector model downloaded successfully!")' > download_models.py
RUN python download_models.py || echo "Failed to download turn detector model during build, will try at runtime"

# Download Silero VAD model at build time
RUN echo 'import sys; sys.path.append("/opt/venv/lib/python3.10/site-packages"); from livekit.plugins import silero; vad = silero.VAD.load(); print("Silero VAD model downloaded successfully!")' > download_vad.py
RUN python download_vad.py || echo "Failed to download VAD model during build, will try at runtime"

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

# Copy downloaded models from builder
COPY --from=builder /root/.cache/livekit-plugins-turn-detector/ /root/.cache/livekit-plugins-turn-detector/
COPY --from=builder /root/.cache/livekit-plugins-silero/ /root/.cache/livekit-plugins-silero/

# Copy necessary files
COPY nhs_agents.py .
COPY download_models.py .

# Create necessary directories
RUN mkdir -p ./data

# Create empty .env file (will be populated via environment variables or volume mount)
RUN touch .env

# Set environment variable for unbuffered output
ENV PYTHONUNBUFFERED=1

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD ps aux | grep python | grep nhs_agents.py || exit 1

# Create startup script that tries to force model download once more at runtime
RUN echo '#!/bin/bash\n\
echo "Ensuring models are downloaded..."\n\
python download_models.py || echo "Some models may not have downloaded correctly"\n\
echo "Running NHS Virtual Assistant..."\n\
exec python nhs_agents.py start' > /app/start.sh && \
    chmod +x /app/start.sh

# Run the startup script
CMD ["/app/start.sh"] 