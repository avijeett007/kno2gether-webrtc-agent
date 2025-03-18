# Memory-Enabled Mental Health Coach Voice Agent

This project implements a mental health coach voice agent using LiveKit and AutoGen. The agent uses teachability features from AutoGen to learn from interactions and LlamaIndex for vector-based memory storage of conversations.

## Features

### Memory System
- Uses LlamaIndex for vector-based memory storage of conversations
- Implements TeachableMemory class for storing and retrieving relevant conversation context
- Leverages AutoGen's teachability features to learn from user interactions

### Task Tracking
- Creates SQLite database for tracking mental health goals and tasks
- Implements TaskManager class with methods to add, update, and retrieve tasks
- Adds automatic task extraction from user speech using regex patterns
- Implements progress tracking for tasks mentioned in conversations

### User Insights
- Extracts and stores user insights from conversations
- Categorizes insights into preferences, feelings, challenges, goals, and beliefs
- Uses insights to personalize agent responses

### LiveKit Integration
- Enhances the voice pipeline with callbacks for analyzing user and agent speech
- Uses before_llm_callback to incorporate memory and task context
- Adds async event handlers for user_speech_committed and agent_speech_committed

### Mental Health Coach Persona
- Creates a compassionate mental health coach persona named Sarah
- Includes detailed guidelines for communication style and approach
- Implements personalized welcome message

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
MEMORY_PERSIST_DIR=./data/memory
INDEX_NAME=mental_health_coach
TASK_DB_PATH=./data/tasks.db
MODEL_NAME=gpt-4o-mini
```

3. Run the agent:
```bash
python memoryAgent.py
```

## Architecture

The system consists of several key components:

1. **TeachableMemory**: Handles storage and retrieval of conversation history using vector embeddings
2. **TaskManager**: Manages user tasks and progress tracking
3. **TeachableAgent**: Integrates with AutoGen's teachability features
4. **TeachableVoiceAgentSystem**: Combines all components and manages the conversation flow
5. **LiveKit Integration**: Handles voice pipeline and callbacks

## Customization

You can customize the agent by:

1. Modifying the system prompt in the `entrypoint` function
2. Adjusting the regex patterns for task and insight extraction
3. Changing the TTS voice and emotion settings
4. Updating the welcome message

## License

This project is licensed under the MIT License - see the LICENSE file for details.
