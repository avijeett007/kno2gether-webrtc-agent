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
    
    def process_user_message(self, message: str):
        """Process a user message for memory storage"""
        try:
            # Store in memory
            self.memory.add_to_memory(
                text=message, 
                metadata={"role": "user"}
            )
            
        except Exception as e:
            logger.error(f"Error processing user message: {e}")
    
    async def process_agent_response(self, assistant_text: str):
        """Process agent response for memory and insights"""
        try:
            # Only process if we have a user message to pair it with
            if not self.last_user_message:
                return
            
            # Store conversation in memory
            conversation = f"User: {self.last_user_message}\nAssistant: {assistant_text}"
            self.memory.add_to_memory(
                text=conversation,
                metadata={"type": "conversation"}
            )
            
            # Extract potential insights
            asyncio.create_task(self._extract_insights_from_conversation(
                user_message=self.last_user_message,
                assistant_message=assistant_text
            ))
            
            # Reset last user message
            self.last_user_message = ""
            
        except Exception as e:
            logger.error(f"Error processing agent response: {e}")
    
    async def _extract_insights_from_conversation(self, user_message: str, assistant_message: str):
        """Extract insights from conversation using pattern matching"""
        try:
            # Simple patterns to extract insights
            patterns = {
                "preference": [r"I (?:like|love|enjoy|prefer) ([^.!?]+)", r"I don't (?:like|love|enjoy) ([^.!?]+)"],
                "feeling": [r"I (?:feel|am feeling) ([^.!?]+)", r"I'm (?:feeling|experiencing) ([^.!?]+)"],
                "challenge": [r"I (?:struggle with|find it hard to) ([^.!?]+)", r"It's (?:difficult|hard) for me to ([^.!?]+)"],
                "goal": [r"I (?:want to|would like to|hope to) ([^.!?]+)", r"My goal is to ([^.!?]+)"],
                "belief": [r"I (?:believe|think) that ([^.!?]+)", r"I'm convinced that ([^.!?]+)"]
            }
            
            # Extract insights from user message
            for category, category_patterns in patterns.items():
                for pattern in category_patterns:
                    matches = re.finditer(pattern, user_message, re.IGNORECASE)
                    for match in matches:
                        insight = match.group(1).strip()
                        if insight and len(insight) > 5:
                            self.task_db.add_insight(insight, category)
            
            # Extract potential tasks from user message
            task_patterns = [
                r"I need to ([^.!?]+)",
                r"I should ([^.!?]+)",
                r"I have to ([^.!?]+)",
                r"I'm going to ([^.!?]+)",
                r"I plan to ([^.!?]+)"
            ]
            
            for pattern in task_patterns:
                matches = re.finditer(pattern, user_message, re.IGNORECASE)
                for match in matches:
                    task = match.group(1).strip()
                    if task and len(task) > 5:
                        self.task_db.add_task(task)
            
        except Exception as e:
            logger.error(f"Error extracting insights: {e}")

def prewarm(proc: JobProcess):
    """Preload models for faster startup"""
    # Load VAD model for voice activity detection
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("Prewarmed models loaded")

async def entrypoint(ctx: JobContext):
    """Main entry point for the LiveKit agent"""
    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    # Initialize memory agent
    memory_agent = MemoryAgent()
    
    # Create initial chat context
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are Sarah, a mental health coach with expertise in psychological support. "
            "Your goal is to support users in their mental wellbeing journey. "
            "Follow these guidelines:\n"
            "1. Be empathetic and understanding\n"
            "2. Ask open-ended questions to encourage reflection\n"
            "3. Offer practical suggestions when appropriate\n"
            "4. Acknowledge and validate feelings\n"
            "5. Maintain a positive and supportive tone\n"
            "6. Reference past conversations when relevant\n"
            "7. Help users track their mental health goals\n"
            "8. Encourage healthy habits and coping strategies\n\n"
            "You have access to the user's previous conversations and can use this to personalize your responses."
        ),
    )
    
    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"Starting memory-enabled mental health coach for participant {participant.identity}")
    
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
        llm=openai.LLM(model=MODEL_NAME),
        tts=cartesia_tts.TTS(
            model="sonic",
            voice="c2ac25f9-ecc4-4f56-9095-651354df60c0",
            emotion=["curiosity:highest", "positivity:high"]
        ),
        chat_ctx=initial_ctx,
        fnc_ctx=memory_agent.function_context,
        turn_detector=turn_detector.EOUModel(),
        before_llm_cb=memory_agent.before_llm_callback,
    )
    
    # Set up synchronous event handlers
    @agent.on("user_speech_committed")
    def on_user_speech_committed(msg: llm.ChatMessage):
        if isinstance(msg.content, list):
            content = "\n".join(
                "[image]" if isinstance(x, llm.ChatImage) else str(x) for x in msg.content
            )
        else:
            content = msg.content
            
        logger.info(f"User speech committed: {content[:50]}...")
        memory_agent.process_user_message(content)
    
    @agent.on("agent_speech_committed")
    def on_agent_speech_committed(msg: llm.ChatMessage):
        logger.info(f"Agent speech committed: {msg.content[:50]}...")
        # Use asyncio.create_task to run the async function
        asyncio.create_task(memory_agent.process_agent_response(msg.content))
    
    # Set up metrics collection
    usage_collector = metrics.UsageCollector()
    @agent.on("metrics_collected")
    def on_metrics_collected(mtrcs: metrics.AgentMetrics):
        metrics.log_metrics(mtrcs)
        usage_collector.collect(mtrcs)
    
    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")
    ctx.add_shutdown_callback(log_usage)
    
    # Start the agent
    agent.start(ctx.room, participant)
    
    # Create welcome message based on any existing memory
    pending_tasks = memory_agent.task_db.get_pending_tasks()
    insights = memory_agent.task_db.get_insights()
    
    welcome_message = "Hello! I'm Sarah, your mental health coach. How are you feeling today?"
    
    if pending_tasks or insights:
        # We have previous interaction data, create a more personalized greeting
        welcome_message = "Welcome back! I'm Sarah, your mental health coach. How have you been since our last conversation?"
    
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
