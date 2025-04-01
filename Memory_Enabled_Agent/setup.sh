#!/bin/bash

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed. Please install Python 3 and try again."
    exit 1
fi

# Set up virtual environment
echo "Setting up virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if Cerebras SDK was installed successfully
if pip show cerebras_cloud_sdk &> /dev/null; then
    echo "Cerebras SDK installed successfully!"
else
    echo "Warning: Cerebras SDK installation may have failed. Will fall back to OpenAI for knowledge base selection."
fi

# Print environment setup instructions
echo ""
echo "Setup completed successfully!"
echo ""
echo "To run the NHS agent, you need to set the following environment variables:"
echo "  - OPENAI_API_KEY: Required for embeddings and fallback LLM"
echo "  - CEREBRAS_API_KEY: Optional but recommended for cost-efficient knowledge base selection"
echo "  - QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY: Required for vector storage"
echo ""
echo "You can set these in a .env file or export them in your environment."
echo ""
echo "To activate the virtual environment and run the agent:"
echo "  source venv/bin/activate"
echo "  python nhs_agents.py start"
echo "" 