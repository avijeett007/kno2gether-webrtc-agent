#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}NHS Virtual Assistant - Setup Test${NC}"
echo "====================================="

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker is installed${NC}"

# Check for Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker Compose is installed${NC}"

# Check for .env file
if [ ! -f .env ]; then
    echo -e "${YELLOW}WARNING: .env file not found.${NC}"
    echo "Creating from template..."
    cp .env.template .env
    echo -e "${YELLOW}Please edit the .env file with your actual credentials.${NC}"
else
    echo -e "${GREEN}✓ .env file exists${NC}"
fi

# Create directories
mkdir -p data
mkdir -p model_cache
echo -e "${GREEN}✓ Created data and model_cache directories${NC}"

# Check for the main Python file
if [ ! -f nhs_agents.py ]; then
    echo -e "${RED}ERROR: nhs_agents.py not found.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ nhs_agents.py exists${NC}"

# Check for requirements.txt
if [ ! -f requirements.txt ]; then
    echo -e "${RED}ERROR: requirements.txt not found.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ requirements.txt exists${NC}"

echo -e "\n${GREEN}All checks passed!${NC}"
echo -e "You can now build and run the NHS Virtual Assistant with:"
echo -e "${YELLOW}./run_docker.sh build${NC}"
echo -e "${YELLOW}./run_docker.sh start${NC}"

echo -e "\nTo verify the container is running correctly:"
echo -e "${YELLOW}./run_docker.sh status${NC}"

chmod +x run_docker.sh
echo -e "\n${GREEN}Made run_docker.sh executable${NC}" 