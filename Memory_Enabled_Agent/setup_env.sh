#!/bin/bash

# Script to generate a .env file from the template
# Author: Avijit Sarkar

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}NHS Virtual Assistant - Environment Setup${NC}"
echo "====================================="

# Check if .env already exists
if [ -f .env ]; then
    echo -e "${YELLOW}WARNING: An .env file already exists.${NC}"
    read -p "Do you want to overwrite it? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup canceled. Existing .env file was not modified."
        exit 0
    fi
fi

# Create .env from template
cp .env.template .env

echo -e "${GREEN}Created .env file from template.${NC}"
echo "Now let's configure your environment variables:"

# LiveKit configuration
echo -e "\n${YELLOW}LiveKit Configuration:${NC}"
read -p "Enter your LiveKit API Key: " livekit_api_key
read -p "Enter your LiveKit API Secret: " livekit_api_secret
read -p "Enter your LiveKit URL (default: wss://your-livekit-server.com): " livekit_url
livekit_url=${livekit_url:-wss://your-livekit-server.com}

# OpenAI configuration
echo -e "\n${YELLOW}OpenAI Configuration:${NC}"
read -p "Enter your OpenAI API Key: " openai_api_key

# Qdrant configuration
echo -e "\n${YELLOW}Qdrant Configuration:${NC}"
read -p "Enter your Qdrant Host: " qdrant_host
read -p "Enter your Qdrant Port (default: 6333): " qdrant_port
qdrant_port=${qdrant_port:-6333}
read -p "Enter your Qdrant API Key: " qdrant_api_key
read -p "Use TLS for Qdrant? (true/false, default: true): " qdrant_tls
qdrant_tls=${qdrant_tls:-true}

# Model configuration
echo -e "\n${YELLOW}Model Configuration:${NC}"
read -p "Enter the model name (default: gpt-4o-mini): " model_name
model_name=${model_name:-gpt-4o-mini}

# Update .env file
sed -i.bak "s|LIVEKIT_API_KEY=.*|LIVEKIT_API_KEY=$livekit_api_key|g" .env
sed -i.bak "s|LIVEKIT_API_SECRET=.*|LIVEKIT_API_SECRET=$livekit_api_secret|g" .env
sed -i.bak "s|LIVEKIT_URL=.*|LIVEKIT_URL=$livekit_url|g" .env
sed -i.bak "s|OPENAI_API_KEY=.*|OPENAI_API_KEY=$openai_api_key|g" .env
sed -i.bak "s|MODEL_NAME=.*|MODEL_NAME=$model_name|g" .env
sed -i.bak "s|QDRANT_HOST=.*|QDRANT_HOST=$qdrant_host|g" .env
sed -i.bak "s|QDRANT_PORT=.*|QDRANT_PORT=$qdrant_port|g" .env
sed -i.bak "s|QDRANT_API_KEY=.*|QDRANT_API_KEY=$qdrant_api_key|g" .env
sed -i.bak "s|QDRANT_TLS=.*|QDRANT_TLS=$qdrant_tls|g" .env

# Remove backup file
rm -f .env.bak

echo -e "\n${GREEN}Environment setup complete!${NC}"
echo "Your .env file has been created and configured."
echo -e "You can now run ${YELLOW}./run_docker.sh build${NC} to build the Docker image." 