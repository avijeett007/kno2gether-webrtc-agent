#!/bin/bash

# Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is required but not installed. Please install Conda and try again."
    exit 1
fi

# Ask which method to use for environment setup
echo "Choose environment setup method:"
echo "1) Use pip to install packages (faster)"
echo "2) Use conda environment.yml (more reliable but slower)"
read -p "Enter option (1 or 2): " env_method

# Setup Conda environment based on chosen method
if [[ $env_method == "2" ]]; then
    # Use environment.yml
    echo "Setting up Conda environment from environment.yml..."
    if conda env list | grep -q "memoryagent"; then
        read -p "Environment 'memoryagent' already exists. Do you want to update it? (y/n): " update_env
        if [[ $update_env == "y" || $update_env == "Y" ]]; then
            conda env update -f environment.yml
        fi
    else
        conda env create -f environment.yml
    fi
else
    # Use pip (default)
    # Check if memoryagent environment exists, create if not
    if ! conda env list | grep -q "memoryagent"; then
        echo "Creating memoryagent Conda environment..."
        conda create -y -n memoryagent python=3.10
    else
        echo "Using existing memoryagent Conda environment..."
    fi
    
    # Activate the environment
    eval "$(conda shell.bash hook)"
    conda activate memoryagent
    
    # Install requirements
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate memoryagent

# Check if Cerebras SDK was installed successfully
if pip show cerebras_cloud_sdk &> /dev/null; then
    echo "Cerebras SDK installed successfully!"
else
    echo "Warning: Cerebras SDK installation may have failed. Will fall back to OpenAI for knowledge base selection."
fi

# Check if Supabase client was installed successfully
if pip show supabase &> /dev/null; then
    echo "Supabase client installed successfully!"
else
    echo "Warning: Supabase client installation may have failed. Goals tracking and customizable settings will be unavailable."
fi

# Download required models
echo "Downloading required models..."
python download_models.py
if [ $? -eq 0 ]; then
    echo "✅ Models downloaded successfully!"
else
    echo "⚠️ Some models failed to download. The agent may still work but with reduced functionality."
fi

# Ask if user wants to set up Supabase tables
read -p "Do you want to set up Supabase tables for goals tracking and agent settings? (y/n): " setup_supabase
if [[ $setup_supabase == "y" || $setup_supabase == "Y" ]]; then
    echo ""
    echo "NOTE: Supabase tables must be created manually via the Supabase dashboard"
    echo "before running this setup. Please follow the instructions in README.md"
    echo "under 'Supabase Setup (for OYOS Agent)' section."
    echo ""
    read -p "Have you already created the required tables in Supabase? (y/n): " tables_created
    if [[ $tables_created != "y" && $tables_created != "Y" ]]; then
        echo "Please create the tables first and then run this setup again."
        echo "See README.md for detailed instructions."
    else
        # Get Supabase credentials
        read -p "Enter your Supabase URL: " supabase_url
        read -p "Enter your Supabase API key: " supabase_key
        
        if [ -n "$supabase_url" ] && [ -n "$supabase_key" ]; then
            echo "Setting up Supabase tables..."
            python setup_supabase.py --url "$supabase_url" --key "$supabase_key"
            if [ $? -eq 0 ]; then
                echo "✅ Supabase settings configured successfully!"
            else
                echo "⚠️ Failed to configure Supabase settings. Check the logs for details."
                echo "If the tables don't exist, please create them manually via the Supabase dashboard."
                echo "See README.md for detailed instructions."
            fi
        else
            echo "Supabase URL and API key are required for table setup. Skipping."
        fi
    fi
fi

# Print environment setup instructions
echo ""
echo "Setup completed successfully!"
echo ""
echo "To run the OYOS coach/DISC agent, you need to set the following environment variables:"
echo "  - OPENAI_API_KEY: Required for embeddings and LLM"
echo "  - CEREBRAS_API_KEY: Optional but recommended for cost-efficient AI processing"
echo "  - SUPABASE_URL, SUPABASE_KEY: Optional for goals tracking and customizable settings"
echo "  - QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY: Required for vector storage memory"
echo ""
echo "You can set these in a .env file or export them in your environment."
echo ""
echo "To activate the Conda environment and run the agent:"
echo "  conda activate memoryagent"
echo "  python oyos_coach_disc_agent.py start"
echo "" 