# NHS Virtual Assistant - Docker Setup

This directory contains Docker configuration for running the NHS Virtual Assistant agent.

## Prerequisites

- Docker and Docker Compose installed on your system
- LiveKit server (for WebRTC audio/video communication)
- Qdrant instance (for vector storage)
- OpenAI API key

## Setup

1. **Create your environment file**

   Copy the template environment file and fill in your credentials:

   ```bash
   cp .env.template .env
   ```

   Edit the `.env` file with your actual credentials:
   - LiveKit API key and secret
   - OpenAI API key
   - Qdrant connection details

   The Docker Compose setup will load all variables from the `.env` file automatically.

2. **Build and run with Docker Compose**

   ```bash
   docker-compose up --build
   ```

   This will:
   - Build the Docker image with all required dependencies
   - Start the NHS agent service
   - Mount the `./data` directory for persistent storage

3. **For production deployment**

   For production, you may want to run in detached mode:

   ```bash
   docker-compose up -d
   ```

   To view logs:

   ```bash
   docker-compose logs -f
   ```

## Configuration Options

The agent can be configured through environment variables in the `.env` file:

- `MODEL_NAME`: The OpenAI model to use (default: gpt-4o-mini)
- `QDRANT_TLS`: Whether to use TLS for Qdrant connection (default: true)
- `LOG_LEVEL`: Logging level (default: INFO)

You can add any other environment variables needed by the application to the `.env` file and they will be automatically available inside the container.

## Troubleshooting

- **Connection issues**: Ensure your LiveKit server is accessible
- **Memory issues**: Check if your Qdrant instance is properly configured
- **Agent not starting**: Check logs with `docker-compose logs`

## Volumes

- `./data`: Contains persistence data and logs

## Security Notes

- Never commit your `.env` file to version control
- Secure your Qdrant instance with proper authentication
- Set up proper network security for your LiveKit server 