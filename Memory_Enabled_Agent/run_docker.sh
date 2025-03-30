#!/bin/bash

# Run Docker script for NHS Virtual Assistant
# Usage: ./run_docker.sh [command]
# Commands: 
#   build     - Build the Docker image
#   start     - Start the containers
#   stop      - Stop the containers
#   restart   - Restart the containers
#   logs      - View logs
#   shell     - Access shell inside container
#   status    - View container status
#   download-models - Download models for NHS Virtual Assistant

# Default command
COMMAND=${1:-start}

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Environment file (.env) not found. Creating from template..."
    cp .env.template .env
    echo "Please edit .env with your actual credentials before continuing."
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p data

# Execute command
case $COMMAND in
    build)
        echo "Building NHS Virtual Assistant Docker image..."
        docker-compose build
        ;;
    start)
        echo "Starting NHS Virtual Assistant..."
        docker-compose up -d
        echo "Started. Use './run_docker.sh logs' to view logs."
        ;;
    stop)
        echo "Stopping NHS Virtual Assistant..."
        docker-compose down
        ;;
    restart)
        echo "Restarting NHS Virtual Assistant..."
        docker-compose restart
        ;;
    logs)
        echo "Viewing logs (Ctrl+C to exit)..."
        docker-compose logs -f
        ;;
    shell)
        echo "Opening shell in the container..."
        docker-compose exec nhs-agent bash
        ;;
    status)
        echo "Container status:"
        docker-compose ps
        ;;
    download-models)
        echo "Downloading models for NHS Virtual Assistant..."
        docker-compose exec nhs-agent python nhs_agents.py download-files || echo "Error: Is the container running? Try './run_docker.sh start' first."
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo "Usage: ./run_docker.sh [build|start|stop|restart|logs|shell|status|download-models]"
        exit 1
        ;;
esac

exit 0 