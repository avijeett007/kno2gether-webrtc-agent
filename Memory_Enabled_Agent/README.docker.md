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
   - Automatically download required LiveKit models

   For convenience, you can also use the provided script:

   ```bash
   ./run_docker.sh build   # Build the image
   ./run_docker.sh start   # Start the container
   ```

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

## Using the run_docker.sh Script

For ease of use, a utility script `run_docker.sh` is provided with the following commands:

- `./run_docker.sh build` - Build the Docker image
- `./run_docker.sh start` - Start the containers in detached mode
- `./run_docker.sh stop` - Stop the containers
- `./run_docker.sh restart` - Restart the containers
- `./run_docker.sh logs` - View container logs
- `./run_docker.sh shell` - Access a shell inside the container
- `./run_docker.sh status` - Check container status
- `./run_docker.sh download-models` - Manually download LiveKit models

## Model Management

The NHS agent requires specific models for voice activity detection (VAD) and turn detection. These models are automatically downloaded when the container starts up using the `download-files` command built into the agent.

The download process happens automatically when the container starts, but if you need to manually trigger it again, you can do so by:

```bash
docker-compose exec nhs-agent python nhs_agents.py download-files
```

Models are stored in the container's `/app/model_cache` directory, and this directory is mounted as a volume to preserve models between container restarts.

## Implementation Details

This Docker setup follows the same approach used by the Knotie AI Pro Agent:
- Uses a multi-stage build to create a smaller final image
- Downloads models before starting the agent
- Mounts the .env file and model cache as volumes
- Includes a health check to monitor the agent's status 