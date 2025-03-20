"""
Simple Memory-Enabled Mental Health Coach Voice Agent

This module implements a mental health coach voice agent using LiveKit with:
- Vector-based memory storage using LlamaIndex
- Task tracking with SQLite
- Function calling for task and memory management
- Before_llm_callback for context integration

Author: Avijit Sarkar
"""

import os
import re
import sqlite3
import asyncio
import logging
import datetime
import json
from typing import List, Dict, Any, Optional, Annotated

# Load environment variables
from dotenv import load_dotenv

# LiveKit imports
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    WorkerType,
    cli,
    llm,
    metrics,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero, turn_detector
from livekit.plugins.cartesia import tts as cartesia_tts

# LlamaIndex imports
from llama_index.core import (
    Document,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)
from llama_index.embeddings.openai import OpenAIEmbedding

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memory_agent")

# Load environment variables
load_dotenv()

# Environment settings
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

# Room validation settings
ROOM_SUFFIX = "-knomind"
ROOM_VALIDATION_ERROR = "Room name does not match required pattern"

# Memory settings
MEMORY_DIR = "./data/memory"
TASK_DB_PATH = "./data/tasks.db"

# Create necessary directories
os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(os.path.dirname(TASK_DB_PATH), exist_ok=True)

class TaskDB:
    """Simple database for managing mental health tasks and goals"""
    
    def __init__(self, db_path=TASK_DB_PATH):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        """Initialize the database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tasks table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            last_updated TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create insights table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            category TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"Task database initialized at {self.db_path}")
    
    def add_task(self, description: str) -> int:
        """Add a new task"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO tasks (description) VALUES (?)",
            (description,)
        )
        
        task_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Added task: {description[:30]}... (ID: {task_id})")
        return task_id
    
    def update_task(self, task_id: int, status: str) -> bool:
        """Update the status of a task"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.datetime.now().isoformat()
        cursor.execute(
            "UPDATE tasks SET status = ?, last_updated = ? WHERE id = ?",
            (status, now, task_id)
        )
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        if success:
            logger.info(f"Updated task {task_id} status to {status}")
        return success
    
    def get_pending_tasks(self, limit: int = 5) -> List[Dict]:
        """Get pending tasks"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM tasks WHERE status = 'pending' ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        
        tasks = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return tasks
    
    def add_insight(self, content: str, category: str) -> int:
        """Add a new insight about the user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO insights (content, category) VALUES (?, ?)",
            (content, category)
        )
        
        insight_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Added insight: {content[:30]}... (Category: {category})")
        return insight_id
    
    def get_insights(self, category: Optional[str] = None, limit: int = 5) -> List[Dict]:
        """Get insights about the user"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if category:
            cursor.execute(
                "SELECT * FROM insights WHERE category = ? ORDER BY created_at DESC LIMIT ?",
                (category, limit)
            )
        else:
            cursor.execute(
                "SELECT * FROM insights ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )
        
        insights = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return insights

class VectorMemory:
    """Vector memory system using LlamaIndex"""
    
    def __init__(self, persist_dir=MEMORY_DIR):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self._init_storage()
    
    def _init_storage(self):
        """Initialize the vector storage"""
        try:
            # Check if we have existing storage
            index_path = os.path.join(self.persist_dir, "index")
            if os.path.exists(index_path):
                # Load existing index
                logger.info(f"Loading existing vector index from {self.persist_dir}")
                
                # Create storage context
                storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
                
                # Load index
                self.index = load_index_from_storage(storage_context)
            else:
                # Create a new index
                logger.info(f"Creating new vector index at {self.persist_dir}")
                
                # Create index with empty document list
                self.index = VectorStoreIndex(
                    [],
                    embed_model=OpenAIEmbedding()
                )
                
                # Save the empty index
                self.index.storage_context.persist(persist_dir=self.persist_dir)
        
        except Exception as e:
            logger.error(f"Error initializing vector storage: {e}")
            
            # Create a fallback empty index
            logger.info("Creating fallback empty index")
            self.index = VectorStoreIndex(
                [],
                embed_model=OpenAIEmbedding()
            )
    
    def add_to_memory(self, text: str, metadata: Dict[str, Any] = None) -> bool:
        """Add text to vector memory"""
        try:
            # Create default metadata if none provided
            if metadata is None:
                metadata = {}
            
            # Add timestamp to metadata
            metadata["timestamp"] = datetime.datetime.now().isoformat()
            
            # Create document
            doc = Document(text=text, metadata=metadata)
            
            # Add to index
            self.index.insert(doc)
            
            # Persist the updated index
            self.index.storage_context.persist(persist_dir=self.persist_dir)
            
            logger.info(f"Added to memory: {text[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error adding to memory: {e}")
            return False
    
    def query_memory(self, query: str, limit: int = 3) -> str:
        """Query memory for relevant information"""
        try:
            # Create query engine
            query_engine = self.index.as_query_engine(similarity_top_k=limit)
            
            # Execute query
            response = query_engine.query(query)
            
            logger.info(f"Retrieved memory for query: {query[:50]}...")
            return response.response
            
        except Exception as e:
            logger.error(f"Error querying memory: {e}")
            return ""

class MentalHealthCoachFunctions(llm.FunctionContext):
    """Function context for mental health coach"""
    
    def __init__(self, task_db: TaskDB):
        super().__init__()
        self.task_db = task_db
    
    @llm.ai_callable(
        description="Add a new mental health task or goal to track"
    )
    async def add_task(
        self,
        description: Annotated[
            str,
            llm.TypeInfo(
                description="Description of the task or goal"
            )
        ]
    ) -> str:
        """Add a new mental health task"""
        try:
            task_id = self.task_db.add_task(description)
            return f"Task added with ID {task_id}: {description}"
        except Exception as e:
            return f"Failed to add task: {str(e)}"
    
    @llm.ai_callable(
        description="Update the status of a task"
    )
    async def update_task(
        self,
        task_id: Annotated[
            int,
            llm.TypeInfo(
                description="ID of the task to update"
            )
        ],
        status: Annotated[
            str,
            llm.TypeInfo(
                description="New status: pending, in_progress, completed"
            )
        ]
    ) -> str:
        """Update the status of a task"""
        try:
            success = self.task_db.update_task(task_id, status)
            if success:
                return f"Task {task_id} updated to status: {status}"
            else:
                return f"Task {task_id} not found"
        except Exception as e:
            return f"Failed to update task: {str(e)}"
    
    @llm.ai_callable(
        description="Get pending mental health tasks"
    )
    async def get_tasks(
        self,
        limit: Annotated[
            int,
            llm.TypeInfo(
                description="Maximum number of tasks to return"
            )
        ] = 5
    ) -> str:
        """Get pending mental health tasks"""
        try:
            tasks = self.task_db.get_pending_tasks(limit)
            if not tasks:
                return "No pending tasks found."
            
            task_strings = []
            for task in tasks:
                task_strings.append(f"ID {task['id']}: {task['description']} (Status: {task['status']})")
            
            return "Pending tasks:\n" + "\n".join(task_strings)
        except Exception as e:
            return f"Failed to get tasks: {str(e)}"
    
    @llm.ai_callable(
        description="Add an insight about the user"
    )
    async def add_insight(
        self,
        content: Annotated[
            str,
            llm.TypeInfo(
                description="Content of the insight"
            )
        ],
        category: Annotated[
            str,
            llm.TypeInfo(
                description="Category of the insight: preference, feeling, challenge, goal, belief"
            )
        ]
    ) -> str:
        """Add an insight about the user"""
        try:
            insight_id = self.task_db.add_insight(content, category)
            return f"Insight added with ID {insight_id}: {content}"
        except Exception as e:
            return f"Failed to add insight: {str(e)}"
    
    @llm.ai_callable(
        description="Get insights about the user"
    )
    async def get_insights(
        self,
        category: Annotated[
            str,
            llm.TypeInfo(
                description="Category to filter by, or 'all' for all categories"
            )
        ] = "all",
        limit: Annotated[
            int,
            llm.TypeInfo(
                description="Maximum number of insights to return"
            )
        ] = 5
    ) -> str:
        """Get insights about the user"""
        try:
            if category.lower() == "all":
                insights = self.task_db.get_insights(limit=limit)
            else:
                insights = self.task_db.get_insights(category=category, limit=limit)
            
            if not insights:
                return f"No insights found for category '{category}'."
            
            insight_strings = []
            for insight in insights:
                insight_strings.append(f"- {insight['content']} (Category: {insight['category']})")
            
            return "User insights:\n" + "\n".join(insight_strings)
        except Exception as e:
            return f"Failed to get insights: {str(e)}"

class MemoryAgent:
    """Memory-enabled mental health coach agent"""
    
    def __init__(self):
        # Initialize task database
        self.task_db = TaskDB()
        
        # Initialize vector memory
        self.memory = VectorMemory()
        
        # Initialize function context
        self.function_context = MentalHealthCoachFunctions(self.task_db)
        
        # Store last user message for processing
        self.last_user_message = ""
        
        # Track full conversation history
        self.conversation_history = []
        
        logger.info("Memory agent initialized")
    
    async def before_llm_callback(self, assistant: VoicePipelineAgent, chat_ctx: llm.ChatContext):
        """Process context before sending to LLM"""
        try:
            # Get the latest user message
            for msg in reversed(chat_ctx.messages):
                if msg.role == "user" and msg.content:
                    user_message = msg.content
                    if isinstance(user_message, list):
                        # Handle potential image content
                        user_message = "\n".join([
                            str(item) for item in user_message 
                            if not isinstance(item, llm.ChatImage)
                        ])
                    
                    # Store for later use in agent_speech_committed
                    self.last_user_message = user_message
                    
                    # Retrieve memory context
                    memory_context = self.memory.query_memory(user_message)
                    
                    # Get pending tasks
                    tasks = self.task_db.get_pending_tasks()
                    task_context = ""
                    if tasks:
                        task_lines = [f"- {task['description']}" for task in tasks]
                        task_context = "Current tasks to follow up on:\n" + "\n".join(task_lines)
                    
                    # Get insights
                    insights = self.task_db.get_insights()
                    insight_context = ""
                    if insights:
                        insight_lines = [f"- {insight['category']}: {insight['content']}" for insight in insights]
                        insight_context = "Things I know about the user:\n" + "\n".join(insight_lines)
                    
                    # Build context
                    context_parts = []
                    if memory_context:
                        context_parts.append(f"Previous conversation context:\n{memory_context}")
                    if task_context:
                        context_parts.append(task_context)
                    if insight_context:
                        context_parts.append(insight_context)
                    
                    if context_parts:
                        # Create context message
                        context_text = "Here is important context for this conversation:\n\n" + "\n\n".join(context_parts)
                        
                        # Add as system message before the user message
                        for i in range(len(chat_ctx.messages)):
                            if chat_ctx.messages[i].role == "user" and chat_ctx.messages[i].content == user_message:
                                # Insert context message before this user message
                                chat_ctx.messages.insert(i, llm.ChatMessage(
                                    role="system",
                                    content=context_text
                                ))
                                logger.info("Added memory context to chat")
                                break
                    
                    break
            
            # Limit context length to avoid token issues
            if len(chat_ctx.messages) > 15:
                # Keep system messages and most recent messages
                system_messages = [msg for msg in chat_ctx.messages if msg.role == "system"]
                recent_messages = chat_ctx.messages[-12:]  # Keep last 12 messages
                
                # Combine them, keeping at most one system message at the start
                if system_messages:
                    chat_ctx.messages = [system_messages[0]] + recent_messages
                else:
                    chat_ctx.messages = recent_messages
                
                logger.info("Truncated chat context to reduce token usage")
            
        except Exception as e:
            logger.error(f"Error in before_llm_callback: {e}")
    
    def add_user_message(self, message: str):
        """Add a user message to the conversation history"""
        try:
            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "text": message,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            # Store for context in the upcoming agent response
            self.last_user_message = message
            
            logger.info(f"Added user message to conversation history: {message[:50]}...")
            
        except Exception as e:
            logger.error(f"Error adding user message to history: {e}")
    
    def add_agent_message(self, message: str):
        """Add an agent message to the conversation history"""
        try:
            # Add to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "text": message,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            logger.info(f"Added agent message to conversation history: {message[:50]}...")
            
        except Exception as e:
            logger.error(f"Error adding agent message to history: {e}")
    
    async def process_conversation(self):
        """Process the entire conversation history at the end of the session"""
        try:
            logger.info("Processing complete conversation...")
            
            # Skip if no conversation happened
            if not self.conversation_history:
                logger.info("No conversation to process")
                return
            
            # Format the conversation for storage and analysis
            conversation_text = ""
            for message in self.conversation_history:
                prefix = "User" if message["role"] == "user" else "Assistant"
                conversation_text += f"{prefix}: {message['text']}\n\n"
            
            # Store the full conversation in memory
            self.memory.add_to_memory(
                text=conversation_text,
                metadata={"type": "full_conversation", 
                         "timestamp": datetime.datetime.now().isoformat()}
            )
            
            # Extract insights and tasks using pattern matching
            await self._extract_all_insights()
            
            # Analyze conversation with external AI (can be added later)
            await self._analyze_conversation_with_ai()
            
            logger.info("Conversation processing complete")
            
        except Exception as e:
            logger.error(f"Error processing conversation: {e}")
    
    async def _extract_all_insights(self):
        """Extract insights from the entire conversation"""
        try:
            # Patterns for insights
            patterns = {
                "preference": [r"I (?:like|love|enjoy|prefer) ([^.!?]+)", r"I don't (?:like|love|enjoy) ([^.!?]+)"],
                "feeling": [r"I (?:feel|am feeling) ([^.!?]+)", r"I'm (?:feeling|experiencing) ([^.!?]+)"],
                "challenge": [r"I (?:struggle with|find it hard to) ([^.!?]+)", r"It's (?:difficult|hard) for me to ([^.!?]+)"],
                "goal": [r"I (?:want to|would like to|hope to) ([^.!?]+)", r"My goal is to ([^.!?]+)"],
                "belief": [r"I (?:believe|think) that ([^.!?]+)", r"I'm convinced that ([^.!?]+)"]
            }
            
            # Task patterns
            task_patterns = [
                r"I need to ([^.!?]+)",
                r"I should ([^.!?]+)",
                r"I have to ([^.!?]+)",
                r"I'm going to ([^.!?]+)",
                r"I plan to ([^.!?]+)"
            ]
            
            # Process all user messages
            for message in self.conversation_history:
                if message["role"] != "user":
                    continue
                
                user_message = message["text"]
                
                # Extract insights
                for category, category_patterns in patterns.items():
                    for pattern in category_patterns:
                        matches = re.finditer(pattern, user_message, re.IGNORECASE)
                        for match in matches:
                            insight = match.group(1).strip()
                            if insight and len(insight) > 5:
                                self.task_db.add_insight(insight, category)
                
                # Extract tasks
                for pattern in task_patterns:
                    matches = re.finditer(pattern, user_message, re.IGNORECASE)
                    for match in matches:
                        task = match.group(1).strip()
                        if task and len(task) > 5:
                            self.task_db.add_task(task)
            
            logger.info("Extracted insights and tasks from conversation")
            
        except Exception as e:
            logger.error(f"Error extracting insights from conversation: {e}")
    
    async def _analyze_conversation_with_ai(self):
        """Analyze the conversation with an external AI API call to extract deeper insights"""
        try:
            # Skip if no conversation happened
            if not self.conversation_history:
                return
            
            # Format the conversation for analysis
            conversation_text = ""
            for message in self.conversation_history:
                prefix = "User" if message["role"] == "user" else "Assistant"
                conversation_text += f"{prefix}: {message['text']}\n\n"
            
            # Call OpenAI to analyze the conversation
            logger.info("Analyzing conversation with OpenAI...")
            
            try:
                # Import OpenAI client
                from openai import OpenAI
                
                # Create client
                openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                
                # Call API
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",  # Use the same model as the agent for consistency
                    messages=[
                        {"role": "system", "content": "You are an expert mental health analyzer. Analyze this mental health conversation and extract key insights about the user. Output should be JSON with these sections: 1) feelings - emotional states expressed; 2) challenges - difficulties mentioned; 3) goals - aspirations or objectives; 4) preferences - likes and dislikes; 5) beliefs - core beliefs or thoughts; 6) tasks - actionable items the user should work on."},
                        {"role": "user", "content": conversation_text}
                    ],
                    response_format={"type": "json_object"}
                )
                
                # Process the analysis results
                analysis = response.choices[0].message.content
                
                # Parse the JSON
                analysis_data = json.loads(analysis)
                
                # Store insights from analysis
                if "feelings" in analysis_data:
                    for feeling in analysis_data["feelings"]:
                        self.task_db.add_insight(feeling, "feeling")
                
                if "challenges" in analysis_data:
                    for challenge in analysis_data["challenges"]:
                        self.task_db.add_insight(challenge, "challenge")
                
                if "goals" in analysis_data:
                    for goal in analysis_data["goals"]:
                        self.task_db.add_insight(goal, "goal")
                
                if "preferences" in analysis_data:
                    for preference in analysis_data["preferences"]:
                        self.task_db.add_insight(preference, "preference")
                
                if "beliefs" in analysis_data:
                    for belief in analysis_data["beliefs"]:
                        self.task_db.add_insight(belief, "belief")
                
                if "tasks" in analysis_data:
                    for task in analysis_data["tasks"]:
                        self.task_db.add_task(task)
                
                # Store the full analysis as a document in vector memory
                self.memory.add_to_memory(
                    text=json.dumps(analysis_data, indent=2),
                    metadata={"type": "conversation_analysis", 
                             "timestamp": datetime.datetime.now().isoformat()}
                )
                
                logger.info("Successfully analyzed conversation with OpenAI")
                
            except ImportError:
                logger.warning("OpenAI client not installed. Skipping AI analysis.")
            except Exception as e:
                logger.error(f"Error in OpenAI analysis: {e}")
                # Continue execution even if OpenAI analysis fails
            
        except Exception as e:
            logger.error(f"Error analyzing conversation with AI: {e}")

def prewarm(proc: JobProcess):
    """Preload models for faster startup"""
    # Load VAD model for voice activity detection
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("Prewarmed models loaded")

def validate_room_name(room_name: str) -> bool:
    """Validate that the room name ends with the required suffix"""
    return room_name.endswith(ROOM_SUFFIX)

async def entrypoint(ctx: JobContext):
    """Main entry point for the LiveKit agent"""
    logger.info(f"Connecting to room {ctx.room.name}")
    
    # Validate room name
    if not validate_room_name(ctx.room.name):
        logger.warning(f"Room name '{ctx.room.name}' does not have the required suffix '{ROOM_SUFFIX}'")
        # Continue anyway as this is just for personalization
    
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    # Initialize memory agent
    memory_agent = MemoryAgent()
    
    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"Starting memory-enabled mental health coach for participant {participant.identity}")
    
    # Get participant metadata
    user_name = ""
    user_goal = ""
    metadata = participant.metadata
    if metadata:
        try:
            user_data = json.loads(metadata)
            logger.info(f"Parsed user data: {user_data}")
            user_name = user_data.get('name', '')
            user_goal = user_data.get('goal', '')
            
            # Add user info as an insight if available
            if user_name:
                memory_agent.task_db.add_insight(f"User's name is {user_name}", "identity")
            if user_goal:
                memory_agent.task_db.add_insight(f"User's goal: {user_goal}", "goal")
                memory_agent.task_db.add_task(user_goal)
        except json.JSONDecodeError:
            logger.error("Failed to parse participant metadata")
    
    # Create initial chat context with personalized system prompt
    system_prompt = (
        f"You are Craig, a mental health coach with expertise in psychological support. "
        f"Your goal is to support users in their mental wellbeing journey. keep it a conversation with the user."
    )
    
    # Add personalization if user data is available
    if user_name:
        system_prompt += f" You're speaking with {user_name}."
    if user_goal:
        system_prompt += f" {user_name if user_name else 'The user'}'s stated goal is: {user_goal}."
    
    system_prompt += (
        f"\n\nIf you are already aware of the user's name, or tasks, or issues or anything else, "
        f"just start the conversation from there and make it a conversation with the user."
        f"learn more about the user, their name, preferences, surroundings, feelings, challenges, goals, and beliefs "
        f"and everything else that will help your in mental health assessment for the user."
        f"You can also use the user's previous conversations and can use this to personalize your responses."
        f"You can suggest tasks, goals, and other things to the user that will help them in their mental health journey. "
        f"and update the task manager and memory accordingly."

        f"\n\nVoice Optimized Communication Guidelines:\n"
        f"- Use short, clear sentences\n"
        f"- Always end sentences with proper punctuation\n"
        f"- Use verbal backchanneling ('mm-hmm', 'I see', 'right', 'got it')\n"
        f"- Never product emojis or other non-text based responses like * or other symbols as this is a voice communication\n"
         f"-Again REITERATING, DON'T PRODUCE ANYTHING OTHER THAN TEXT. NO ASTERICS, SPECIAL SYMBOLS, CHARACTERS TO LIST THINGS. IT SHOULD BE CONVERSATIONAL AND NATURAL OUTPUT ALWAYS.\n"
        f"- Express dates in MM/DD/YYYY format (e.g., 04/20/2023)\n"
        f"- Use two question marks for emphasized questions (e.g., 'How does that make you feel??')\n"
        f"- Avoid using quotation marks unless referring to a specific quote\n"
        f"- Leave a space between URLs/emails and punctuation (e.g., 'Visit our website? ' instead of 'Visit our website?')\n"
        f"- For numbers that should be spelled out, use '<spell>123-456-7890</spell>' tags\n"
        f"- Keep responses concise and conversational\n"
        
        f"\nExample of spelling out numbers:\n"
        f"'You can reach our support line at <spell>800-555-1234</spell> anytime.'\n"

        f"\nFollow these guidelines:\n"
        f"1. Be empathetic and understanding\n"
        f"2. Ask open-ended questions to encourage reflection\n"
        f"3. Offer practical suggestions when appropriate\n"
        f"4. Acknowledge and validate feelings\n"
        f"5. Maintain a positive and supportive tone\n"
        f"6. Reference past conversations when relevant\n"
        f"7. Help users track their mental health goals\n"
        f"8. Encourage healthy habits and coping strategies\n\n"
        f"Most Importantly, support speech normalization and generate responses optimized for voice output. "
        f"Avoid using emojis and other non-text based responses."
        f"You have access to the user's previous conversations and can use this to personalize your responses."

        f"\n\nRemember: \n"
        f"- Insert appropriate pauses at natural breaking points in conversation\n"
        f"- Use a deliberately paced, calm speaking style.\n"
        f"- Keep the conversation naturally flowing\n"
        f"- Use the user's name occasionally\n"
        f"- Use backchanneling to keep the conversation engaging. like 'I see', 'I understand', 'umm-hmm', 'right', 'got it', etc.\n"
        f"- Never mention the background processing\n"
        f"- Listen for behavioral patterns and adapt follow-up questions\n"
        f"- Adapt questions based on responses received\n"
    )
    
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=system_prompt,
    )
    
    # Create the voice pipeline agent
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata.get("vad", silero.VAD.load()),
        stt=deepgram.STT(
            model="nova-2-general",
            interim_results=True,
            smart_format=True,
            punctuate=True,
            language="en-US",
        ),
        llm=openai.LLM(
            model=MODEL_NAME,
        ),
        tts=cartesia_tts.TTS(
            model="sonic-2",
            voice="7e19344f-9f17-47d7-a13a-4366ad06ebf3",
            sample_rate=24000,
            speed="slow",  # Slower for mental health coaching - supported directly
            emotion=["curiosity", "positivity:high","anger:lowest", "sadness:low"],  # Emotions are supported directly
            # The __experimental_controls is handled internally by the API
        ),
        chat_ctx=initial_ctx,
        fnc_ctx=memory_agent.function_context,
        turn_detector=turn_detector.EOUModel(),
        before_llm_cb=memory_agent.before_llm_callback,
    )
    
    # Simpler event handlers that just record messages
    @agent.on("user_speech_committed")
    def on_user_speech_committed(msg: llm.ChatMessage):
        if isinstance(msg.content, list):
            content = "\n".join(
                "[image]" if isinstance(x, llm.ChatImage) else str(x) for x in msg.content
            )
        else:
            content = msg.content
            
        logger.info(f"User speech committed: {content[:50]}...")
        memory_agent.add_user_message(content)
    
    @agent.on("agent_speech_committed")
    def on_agent_speech_committed(msg: llm.ChatMessage):
        content = msg.content
        logger.info(f"Agent speech committed: {content[:50]}...")
        memory_agent.add_agent_message(content)
    
    # Set up metrics collection
    usage_collector = metrics.UsageCollector()
    @agent.on("metrics_collected")
    def on_metrics_collected(mtrcs: metrics.AgentMetrics):
        metrics.log_metrics(mtrcs)
        usage_collector.collect(mtrcs)
    
    # Process conversation at end of session
    async def end_of_session():
        # Process conversation for insights, tasks, etc.
        await memory_agent.process_conversation()
        
        # Log conversation summary
        user_messages = [msg for msg in memory_agent.conversation_history if msg["role"] == "user"]
        agent_messages = [msg for msg in memory_agent.conversation_history if msg["role"] == "assistant"]
        
        if user_messages:
            # Calculate conversation statistics
            conversation_duration = None
            if len(memory_agent.conversation_history) >= 2:
                first_msg_time = datetime.datetime.fromisoformat(memory_agent.conversation_history[0]["timestamp"])
                last_msg_time = datetime.datetime.fromisoformat(memory_agent.conversation_history[-1]["timestamp"])
                conversation_duration = (last_msg_time - first_msg_time).total_seconds()
            
            # Get insights and tasks
            insights = memory_agent.task_db.get_insights()
            tasks = memory_agent.task_db.get_pending_tasks()
            
            # Log summary
            logger.info("=" * 50)
            logger.info("CONVERSATION SUMMARY")
            logger.info("=" * 50)
            logger.info(f"Total messages: {len(memory_agent.conversation_history)}")
            logger.info(f"User messages: {len(user_messages)}")
            logger.info(f"Agent messages: {len(agent_messages)}")
            
            if conversation_duration:
                minutes = int(conversation_duration // 60)
                seconds = int(conversation_duration % 60)
                logger.info(f"Conversation duration: {minutes}m {seconds}s")
            
            logger.info(f"Insights extracted: {len(insights)}")
            logger.info(f"Tasks identified: {len(tasks)}")
            logger.info("=" * 50)
        
        # Log usage metrics
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")
    
    # Add to shutdown callbacks
    ctx.add_shutdown_callback(end_of_session)
    
    # Start the agent
    agent.start(ctx.room, participant)
    
    # Create welcome message based on any existing memory and user data
    pending_tasks = memory_agent.task_db.get_pending_tasks()
    insights = memory_agent.task_db.get_insights()
    
    # Track if this is a first-time user or returning user
    # If we only have the insights/tasks we just added from metadata, it's a first-time user
    is_first_time = True
    
    # Check if we have more insights than just the ones we added from metadata
    metadata_insight_count = 0
    if user_name:
        metadata_insight_count += 1  # We added name as insight
    if user_goal:
        metadata_insight_count += 1  # We added goal as insight
        # We also added goal as a task, so count that too
    
    # If we have more insights or tasks than we just added, it's a returning user
    if len(insights) > metadata_insight_count or len(pending_tasks) > (1 if user_goal else 0):
        is_first_time = False
    
    # Default welcome message for first time user
    welcome_message = "Hello! I'm Craig, your mental health coach. How are you feeling today?"
    
    # Personalize based on user data from metadata
    if user_name and is_first_time:
        welcome_message = f"Hello {user_name}! I'm Craig, your mental health coach. How are you feeling today?"
    
    # Only show goal in first message for first-time users
    if user_goal and is_first_time:
        welcome_message += f" I understand {user_goal} is something which brings you here today. Let's work on that together."
    
    # Further personalize if we have previous interaction data (returning user)
    if not is_first_time:
        if user_name:
            welcome_message = f"Welcome back, {user_name}! I'm Craig, your mental health coach. How have you been since our last conversation?"
        else:
            welcome_message = "Welcome back! I'm Craig, your mental health coach. How have you been since our last conversation?"
    
    # Send welcome message
    await agent.say(welcome_message, allow_interruptions=True)

if __name__ == "__main__":
    # Run the LiveKit agent
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            worker_type=WorkerType.ROOM,
            num_idle_processes=2  # Keep 2 processes warm for better response time
        ),
    )
