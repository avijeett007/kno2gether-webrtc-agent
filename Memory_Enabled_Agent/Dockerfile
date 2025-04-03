# Build stage
# Note: Docker uses standard Python environment, not Conda which is used for local development
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
COPY oyos_coach_disc_agent.py .
COPY download_models.py .
COPY setup_supabase.py .

# Create necessary directories
RUN mkdir -p ./data

# Create empty .env file (will be populated via environment variables or volume mount)
RUN touch .env

# Set environment variable for unbuffered output
ENV PYTHONUNBUFFERED=1

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD ps aux | grep python | grep oyos_coach_disc_agent.py || exit 1

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Downloading required models (if not already downloaded)..."\n\
python download_models.py || echo "WARNING: Model download failed, continuing anyway"\n\
\n\
# Setup Supabase tables if credentials are provided\n\
if [ -n "$SUPABASE_URL" ] && [ -n "$SUPABASE_KEY" ]; then\n\
    echo "Setting up Supabase tables..."\n\
    python setup_supabase.py --url "$SUPABASE_URL" --key "$SUPABASE_KEY" || echo "WARNING: Supabase setup failed, continuing anyway"\n\
fi\n\
\n\
echo "Starting OYOS Voice AI Agent..."\n\
exec python oyos_coach_disc_agent.py start' > /app/start.sh && \
    chmod +x /app/start.sh

# Run the startup script
CMD ["/app/start.sh"] 