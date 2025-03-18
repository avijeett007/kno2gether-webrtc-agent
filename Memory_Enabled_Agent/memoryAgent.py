"""
Memory-Enabled Mental Health Coach Voice Agent

This module implements a mental health coach voice agent using LiveKit and AutoGen.
The agent uses teachability features from AutoGen to learn from interactions and 
LlamaIndex for vector-based memory storage of conversations.

Key features:
- Memory system for storing and retrieving conversation context
- Task tracking for mental health goals
- User insights extraction and storage
- LiveKit voice pipeline integration
- Mental health coach persona

Author: Avijit Sarkar
"""

import os
import re
import uuid
import json
import logging
import sqlite3
import asyncio
from typing import Dict, List, Optional, Any, AsyncIterable
from datetime import datetime

# AutoGen imports
from autogen import ConversableAgent, config_list_from_json
from autogen.agentchat.contrib.capabilities.teachability import Teachability

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.embeddings import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

# LiveKit imports
import livekit
from livekit import rtc
from livekit.agents import llm, metrics, turn_detector
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import cartesia as cartesia_tts
from livekit.plugins import deepgram, silero, openai as lk_openai
from livekit.plugins.cartesia import TTS
from livekit.rtc import Room, RoomEvent, Participant, Track, TrackEvent
from livekit.agents.job import JobContext, JobProcess, AutoSubscribe

# Load environment variables
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memory_agent")

# Environment settings
OAI_CONFIG_LIST = os.environ.get("OAI_CONFIG_LIST", "oai_config_list.json")
KEY_LOC = os.environ.get("KEY_LOC", ".")
MEMORY_PERSIST_DIR = os.environ.get("MEMORY_PERSIST_DIR", "./memory_storage")
INDEX_NAME = os.environ.get("INDEX_NAME", "teachable_agent_memory")
TASK_DB_PATH = os.environ.get("TASK_DB_PATH", "./data/tasks.sqlite3")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

# Create necessary directories
os.makedirs(os.path.dirname(MEMORY_PERSIST_DIR), exist_ok=True)
os.makedirs(os.path.dirname(TASK_DB_PATH), exist_ok=True)

class TeachableMemory:
    """Memory system for storing and retrieving conversation context"""
    
    def __init__(
        self, 
        persist_directory: str = MEMORY_PERSIST_DIR,
        index_name: str = INDEX_NAME,
        verbose: bool = False
    ):
        self.verbose = verbose
        self.persist_directory = persist_directory
        self.index_name = index_name
        
        # Create the storage directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize the vector store
        self.vector_store = self._initialize_vector_store()
        
        # Initialize insights database
        self.insights_db_path = os.path.join(persist_directory, "insights.db")
        self._initialize_insights_db()
        
        if self.verbose:
            logger.info(f"TeachableMemory initialized with storage at {persist_directory}")
    
    def _initialize_vector_store(self):
        """Initialize the vector store for memory storage"""
        try:
            # Create embeddings model
            embed_model = OpenAIEmbedding()
            
            # Create vector store
            vector_store = QdrantVectorStore(
                collection_name=self.index_name,
                path=self.persist_directory
            )
            
            return vector_store
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            return None
    
    def _initialize_insights_db(self):
        """Initialize the SQLite database for user insights"""
        try:
            conn = sqlite3.connect(self.insights_db_path)
            cursor = conn.cursor()
            
            # Create insights table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS insights (
                    id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    insight TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing insights database: {e}")
    
    async def add_to_memory(self, user_input: str, agent_response: str):
        """Add a conversation exchange to memory"""
        try:
            if not self.vector_store:
                logger.error("Vector store not initialized")
                return
                
            # Create a document with the exchange
            exchange = f"User: {user_input}\nAssistant: {agent_response}"
            
            # Create metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "type": "conversation",
                "user_input": user_input,
                "agent_response": agent_response
            }
            
            # Create document
            doc = Document(text=exchange, metadata=metadata)
            
            # Create index
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            index = VectorStoreIndex.from_documents(
                [doc], 
                storage_context=storage_context,
                embed_model=OpenAIEmbedding()
            )
            
            if self.verbose:
                logger.info(f"Added exchange to memory: {exchange[:50]}...")
                
        except Exception as e:
            logger.error(f"Error adding to memory: {e}")
    
    async def retrieve_relevant_context(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant context from memory based on query"""
        try:
            if not self.vector_store:
                logger.error("Vector store not initialized")
                return ""
                
            # Create retriever
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                embed_model=OpenAIEmbedding()
            )
            
            # Create retriever
            retriever = index.as_retriever(similarity_top_k=top_k)
            
            # Retrieve nodes
            nodes = retriever.retrieve(query)
            
            if not nodes:
                return ""
                
            # Format context
            context = "\n\n".join([node.node.text for node in nodes])
            
            if self.verbose:
                logger.info(f"Retrieved {len(nodes)} relevant memory nodes")
                
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving from memory: {e}")
            return ""
    
    async def add_user_insight(self, category: str, insight: str, confidence: float = 0.7):
        """Add a user insight to the database"""
        try:
            conn = sqlite3.connect(self.insights_db_path)
            cursor = conn.cursor()
            
            # Generate unique ID
            insight_id = str(uuid.uuid4())
            
            # Get current timestamp
            now = datetime.now().isoformat()
            
            # Insert insight
            cursor.execute(
                '''
                INSERT INTO insights (id, category, insight, confidence, created_at)
                VALUES (?, ?, ?, ?, ?)
                ''',
                (insight_id, category, insight, confidence, now)
            )
            
            conn.commit()
            conn.close()
            
            if self.verbose:
                logger.info(f"Added user insight: {category} - {insight}")
                
        except Exception as e:
            logger.error(f"Error adding user insight: {e}")
    
    async def get_recent_insights(self, category: str = None, limit: int = 5) -> List[Dict]:
        """Get recent user insights, optionally filtered by category"""
        try:
            conn = sqlite3.connect(self.insights_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if category:
                cursor.execute(
                    '''
                    SELECT * FROM insights
                    WHERE category = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    ''',
                    (category, limit)
                )
            else:
                cursor.execute(
                    '''
                    SELECT * FROM insights
                    ORDER BY created_at DESC
                    LIMIT ?
                    ''',
                    (limit,)
                )
                
            rows = cursor.fetchall()
            insights = [dict(row) for row in rows]
            
            conn.close()
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting insights: {e}")
            return []
    
    async def get_insights_by_query(self, query: str, limit: int = 5) -> List[Dict]:
        """Get insights relevant to a query using simple keyword matching"""
        try:
            # Get all insights
            all_insights = await self.get_recent_insights(limit=100)
            
            # Simple relevance scoring based on keyword matching
            scored_insights = []
            for insight in all_insights:
                # Calculate simple relevance score
                score = 0
                for word in query.lower().split():
                    if word in insight["insight"].lower() or word in insight["category"].lower():
                        score += 1
                
                if score > 0:
                    scored_insights.append((score, insight))
            
            # Sort by score and take top results
            scored_insights.sort(reverse=True, key=lambda x: x[0])
            results = [insight for _, insight in scored_insights[:limit]]
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting insights by query: {e}")
            return []


class TaskManager:
    """Manages tasks and goals for the mental health coach agent"""
    
    def __init__(self, db_path: str = TASK_DB_PATH, verbose: bool = False):
        self.db_path = db_path
        self.verbose = verbose
        self.setup_database()
    
    def setup_database(self):
        """Set up SQLite database for task tracking"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tasks table if it doesn't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                status TEXT NOT NULL,
                priority TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                due_date TEXT,
                completion_date TEXT,
                notes TEXT
            )
            ''')
            
            # Create progress table for tracking task progress
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS progress (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                description TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            )
            ''')
            
            # Create user_insights table for storing insights about the user
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_insights (
                id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                insight TEXT NOT NULL,
                confidence REAL,
                recorded_at TEXT NOT NULL
            )
            ''')
            
            # Commit changes and close connection
            conn.commit()
            conn.close()
            
            if self.verbose:
                logger.info("Task database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error setting up task database: {e}")
            raise
    
    async def add_task(self, description: str, priority: str = "medium", due_date: str = None) -> str:
        """Add a new task to the database"""
        try:
            task_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                '''
                INSERT INTO tasks 
                (id, description, status, priority, created_at, updated_at, due_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''',
                (task_id, description, "pending", priority, now, now, due_date)
            )
            
            conn.commit()
            conn.close()
            
            if self.verbose:
                logger.info(f"Added task: {description}")
                
            return task_id
            
        except Exception as e:
            logger.error(f"Error adding task: {e}")
            return None
    
    async def update_task_status(self, task_id: str, status: str) -> bool:
        """Update the status of a task"""
        try:
            now = datetime.now().isoformat()
            completion_date = now if status == "completed" else None
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                '''
                UPDATE tasks 
                SET status = ?, updated_at = ?, completion_date = ?
                WHERE id = ?
                ''',
                (status, now, completion_date, task_id)
            )
            
            success = cursor.rowcount > 0
            
            conn.commit()
            conn.close()
            
            if self.verbose and success:
                logger.info(f"Updated task {task_id} status to {status}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error updating task status: {e}")
            return False
    
    async def add_progress(self, task_id: str, description: str) -> str:
        """Add a progress update for a task"""
        try:
            progress_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                '''
                INSERT INTO progress 
                (id, task_id, description, recorded_at)
                VALUES (?, ?, ?, ?)
                ''',
                (progress_id, task_id, description, now)
            )
            
            conn.commit()
            conn.close()
            
            if self.verbose:
                logger.info(f"Added progress for task {task_id}: {description}")
                
            return progress_id
            
        except Exception as e:
            logger.error(f"Error adding progress: {e}")
            return None
    
    async def add_user_insight(self, category: str, insight: str, confidence: float = 0.8) -> str:
        """Add an insight about the user"""
        try:
            insight_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                '''
                INSERT INTO user_insights 
                (id, category, insight, confidence, recorded_at)
                VALUES (?, ?, ?, ?, ?)
                ''',
                (insight_id, category, insight, confidence, now)
            )
            
            conn.commit()
            conn.close()
            
            if self.verbose:
                logger.info(f"Added user insight ({category}): {insight}")
                
            return insight_id
            
        except Exception as e:
            logger.error(f"Error adding user insight: {e}")
            return None
    
    async def get_pending_tasks(self, limit: int = 5) -> List[Dict]:
        """Get pending tasks"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # This enables column access by name
            cursor = conn.cursor()
            
            cursor.execute(
                '''
                SELECT * FROM tasks 
                WHERE status = 'pending' 
                ORDER BY 
                    CASE priority
                        WHEN 'high' THEN 1
                        WHEN 'medium' THEN 2
                        WHEN 'low' THEN 3
                        ELSE 4
                    END,
                    created_at DESC
                LIMIT ?
                ''',
                (limit,)
            )
            
            rows = cursor.fetchall()
            tasks = [dict(row) for row in rows]
            
            conn.close()
            
            return tasks
            
        except Exception as e:
            logger.error(f"Error getting pending tasks: {e}")
            return []
    
    async def get_recent_progress(self, task_id: str = None, limit: int = 5) -> List[Dict]:
        """Get recent progress updates, optionally filtered by task"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if task_id:
                cursor.execute(
                    '''
                    SELECT p.*, t.description as task_description 
                    FROM progress p
                    JOIN tasks t ON p.task_id = t.id
                    WHERE p.task_id = ?
                    ORDER BY p.recorded_at DESC
                    LIMIT ?
                    ''',
                    (task_id, limit)
                )
            else:
                cursor.execute(
                    '''
                    SELECT p.*, t.description as task_description 
                    FROM progress p
                    JOIN tasks t ON p.task_id = t.id
                    ORDER BY p.recorded_at DESC
                    LIMIT ?
                    ''',
                    (limit,)
                )
            
            rows = cursor.fetchall()
            progress = [dict(row) for row in rows]
            
            conn.close()
            
            return progress
            
        except Exception as e:
            logger.error(f"Error getting progress: {e}")
            return []
    
    async def get_relevant_insights(self, category: str = None, limit: int = 5) -> List[Dict]:
        """Get user insights, optionally filtered by category"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if category:
                cursor.execute(
                    '''
                    SELECT * FROM user_insights
                    WHERE category = ?
                    ORDER BY confidence DESC, recorded_at DESC
                    LIMIT ?
                    ''',
                    (category, limit)
                )
            else:
                cursor.execute(
                    '''
                    SELECT * FROM user_insights
                    ORDER BY confidence DESC, recorded_at DESC
                    LIMIT ?
                    ''',
                    (limit,)
                )
            
            rows = cursor.fetchall()
            insights = [dict(row) for row in rows]
            
            conn.close()
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting insights: {e}")
            return []
    
    async def extract_tasks_from_text(self, text: str) -> List[Dict]:
        """Extract potential tasks from text using regex patterns"""
        task_patterns = [
            r"(?:I (?:will|should|need to|want to|have to)|(?:Let's|Let me)) ([^.!?]+)",
            r"(?:My|The) (?:goal|task|objective|plan) is to ([^.!?]+)",
            r"I'm going to ([^.!?]+)",
            r"(?:I'll|I will) try to ([^.!?]+)"
        ]
        
        potential_tasks = []
        
        for pattern in task_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                task_description = match.group(1).strip()
                if task_description and len(task_description) > 5:
                    potential_tasks.append({
                        "description": task_description,
                        "confidence": 0.8,  # Default confidence
                        "source": text
                    })
        
        return potential_tasks
    
    async def extract_insights_from_text(self, text: str) -> List[Dict]:
        """Extract potential user insights from text using regex patterns"""
        insight_patterns = {
            "preference": [
                r"I (?:like|love|enjoy|prefer) ([^.!?]+)",
                r"I don't (?:like|love|enjoy|prefer) ([^.!?]+)"
            ],
            "feeling": [
                r"I (?:feel|am feeling) ([^.!?]+)",
                r"I'm (?:feeling|experiencing) ([^.!?]+)"
            ],
            "challenge": [
                r"I (?:struggle|have trouble|find it difficult) (?:with|to) ([^.!?]+)",
                r"It's (?:hard|difficult|challenging) for me to ([^.!?]+)"
            ],
            "goal": [
                r"I (?:want|would like) to ([^.!?]+)",
                r"My (?:goal|aim|objective) is to ([^.!?]+)"
            ],
            "belief": [
                r"I (?:believe|think|feel that) ([^.!?]+)",
                r"I'm (?:convinced|sure) that ([^.!?]+)"
            ]
        }
        
        potential_insights = []
        
        for category, patterns in insight_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    insight = match.group(1).strip()
                    if insight and len(insight) > 3:
                        potential_insights.append({
                            "category": category,
                            "insight": insight,
                            "confidence": 0.7,  # Default confidence
                            "source": text
                        })
        
        return potential_insights


class TeachableAgent:
    """AutoGen teachable agent wrapper"""
    
    def __init__(
        self, 
        agent_name: str = "teachable_voice_agent",
        reset_db: bool = False,
        model_name: str = MODEL_NAME,
        verbose: int = 1
    ):
        self.agent_name = agent_name
        self.verbose = verbose
        self.model_name = model_name
        self.teachability = None
        
        # Create teachable agent
        self.teachable_agent = self.create_teachable_agent(reset_db)
        
        if self.verbose:
            logger.info(f"Teachable agent initialized with model {model_name}")
    
    def create_teachable_agent(self, reset_db: bool = False):
        """Create the AutoGen teachable agent"""
        try:
            # Get API key from environment
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            
            if openai_api_key:
                # Create config directly from environment variables
                logger.info("Using OpenAI API key from environment variables")
                config_list = [
                    {
                        "model": self.model_name,
                        "api_key": openai_api_key,
                    }
                ]
            else:
                # Try to load from config file as fallback
                logger.info(f"OpenAI API key not found in environment, trying to load from {OAI_CONFIG_LIST}")
                try:
                    config_list = config_list_from_json(
                        env_or_file=OAI_CONFIG_LIST,
                        filter_dict={"model": [self.model_name]},
                        file_location=KEY_LOC
                    )
                except Exception as e:
                    logger.error(f"Failed to load config from {OAI_CONFIG_LIST}: {e}")
                    raise ValueError(f"OpenAI API key not available from environment or config file.")
            
            logger.info(f"Creating teachable agent with model: {self.model_name}")
            
            # Create base conversable agent
            teachable_agent = ConversableAgent(
                name=self.agent_name,
                llm_config={"config_list": config_list, "timeout": 120, "cache_seed": None},
                system_message=(
                    "You are Sarah, a compassionate mental health coach. "
                    "Your primary goal is to support users through their mental health journey with empathy, "
                    "understanding, and evidence-based techniques. You learn from interactions and remember "
                    "important information about the user to provide personalized support."
                )
            )
            
            # Create data directory if it doesn't exist
            os.makedirs(f"./data/{self.agent_name}_db", exist_ok=True)
            
            # Add teachability capability
            self.teachability = Teachability(
                verbosity=self.verbose,
                reset_db=reset_db,
                path_to_db_dir=f"./data/{self.agent_name}_db",
                recall_threshold=1.5,
            )
            
            # Attach capability to agent
            self.teachability.add_to_agent(teachable_agent)
            
            return teachable_agent
            
        except Exception as e:
            logger.error(f"Error creating teachable agent: {e}")
            raise
    
    async def generate_response(self, prompt: str) -> str:
        """Generate a response using the teachable agent"""
        try:
            # The AutoGen API typically uses message formats
            # Here we're simplifying to get a direct string response
            response = self.teachable_agent.generate_reply({"content": prompt})
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm having trouble processing that request."
    
    async def process_with_memory(self, user_message: str, context: Dict = None) -> str:
        """Process a message with teachability and memory integration"""
        try:
            # Create a message dictionary that AutoGen expects
            message = {"content": user_message, "role": "user"}
            
            # If we have additional context, add it to the message
            if context:
                message["context"] = context
            
            # Generate a reply using the teachable agent
            # This will automatically use the teachability features
            response = self.teachable_agent.generate_reply(message)
            
            # Check if the teachability component wants to store this interaction
            if self.teachability and hasattr(self.teachability, "should_store_interaction"):
                # This is a custom check - in real implementation, you'd use teachability's methods
                if self.teachability.should_store_interaction(user_message, response):
                    logger.info("Storing interaction in teachable agent memory")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing with memory: {e}")
            return "I'm having trouble processing that request with my memory system."


class TeachableVoiceAgentSystem:
    """System that combines teachable agent with voice capabilities"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        
        # Initialize memory system
        self.memory = TeachableMemory(
            persist_directory=MEMORY_PERSIST_DIR,
            index_name=INDEX_NAME,
            verbose=verbose
        )
        
        # Initialize task manager
        self.task_manager = TaskManager(
            db_path=TASK_DB_PATH,
            verbose=verbose
        )
        
        # Initialize teachable agent
        self.agent = TeachableAgent(
            agent_name="sarah_mental_health_coach",
            reset_db=False,  # Set to True to reset the teachable database
            model_name=MODEL_NAME,
            verbose=1 if verbose else 0
        )
        
        # Initialize conversation history
        self.conversation_history = []
        
        if self.verbose:
            logger.info("TeachableVoiceAgentSystem initialized")
    
    async def process_transcription(self, text: str) -> str:
        """Process user input and generate a response"""
        try:
            # Store user message in conversation history
            self.conversation_history.append({"role": "user", "content": text})
            
            # Retrieve relevant context from memory
            memory_context = await self.memory.retrieve_relevant_context(text)
            
            # Get pending tasks for context
            pending_tasks = await self.task_manager.get_pending_tasks(limit=3)
            task_context = ""
            if pending_tasks:
                task_context = "Current pending tasks:\n" + "\n".join(
                    f"- {task['description']} (Priority: {task['priority']})" 
                    for task in pending_tasks
                )
            
            # Get recent insights for context
            user_insights = await self.memory.get_recent_insights(limit=5)
            insight_context = ""
            if user_insights:
                insight_context = "Recent insights about the user:\n" + "\n".join(
                    f"- {insight['category']}: {insight['insight']}" 
                    for insight in user_insights
                )
            
            # Combine all context
            full_context = {
                "memory_context": memory_context,
                "task_context": task_context,
                "insight_context": insight_context,
                "conversation_history": self.conversation_history[-5:]  # Last 5 exchanges
            }
            
            # Process with teachable agent including memory context
            response = await self.agent.process_with_memory(text, context=full_context)
            
            # Store response in conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Store the exchange in memory
            await self.memory.add_to_memory(text, response)
            
            # Analyze user speech for tasks and insights
            await self.analyze_user_speech(text)
            
            # Analyze agent speech for task progress
            await self.analyze_agent_speech(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing transcription: {e}")
            return "I'm sorry, I encountered an issue processing your message."
    
    async def analyze_user_speech(self, text: str):
        """Analyze user speech for tasks and insights"""
        try:
            # Extract tasks using regex patterns
            task_patterns = [
                r"I need to (.+)",
                r"I want to (.+)",
                r"I should (.+)",
                r"I plan to (.+)",
                r"I'm going to (.+)",
                r"My goal is to (.+)",
                r"I have to (.+)"
            ]
            
            for pattern in task_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    task_description = match.group(1).strip()
                    if task_description:
                        # Add task to task manager
                        await self.task_manager.add_task(description=task_description)
                        if self.verbose:
                            logger.info(f"Extracted task from speech: {task_description}")
            
            # Extract insights using patterns
            insight_patterns = {
                "preference": [r"I like (.+)", r"I prefer (.+)", r"I enjoy (.+)"],
                "feeling": [r"I feel (.+)", r"I am (.+)", r"I'm (.+)"],
                "challenge": [r"I struggle with (.+)", r"It's hard for me to (.+)", r"I find it difficult to (.+)"],
                "goal": [r"I want to (.+)", r"I hope to (.+)", r"I aspire to (.+)"],
                "belief": [r"I believe (.+)", r"I think (.+)", r"I know (.+)"]
            }
            
            for category, patterns in insight_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        insight = match.group(1).strip()
                        if insight:
                            # Add insight to memory
                            await self.memory.add_user_insight(
                                category=category,
                                insight=insight,
                                confidence=0.8  # Default confidence
                            )
                            if self.verbose:
                                logger.info(f"Extracted insight from speech: {category} - {insight}")
                        
        except Exception as e:
            logger.error(f"Error analyzing user speech: {e}")
    
    async def analyze_agent_speech(self, text: str):
        """Analyze agent speech for task progress and follow-ups"""
        try:
            # Check for task progress mentions
            pending_tasks = await self.task_manager.get_pending_tasks()
            
            for task in pending_tasks:
                # Check if the agent's response mentions this task
                if task["description"].lower() in text.lower():
                    # Look for progress indicators
                    progress_indicators = [
                        "progress", "working on", "started", "completed", "finished",
                        "done", "accomplished", "achieved", "success"
                    ]
                    
                    for indicator in progress_indicators:
                        if indicator in text.lower():
                            # Add progress update
                            await self.task_manager.add_progress(
                                task_id=task["id"],
                                description=f"Agent mentioned progress: {text[:100]}..."
                            )
                            
                            # Check for completion indicators
                            completion_indicators = ["completed", "finished", "done", "accomplished"]
                            if any(ci in text.lower() for ci in completion_indicators):
                                await self.task_manager.update_task_status(task["id"], "completed")
                            
                            break
                            
        except Exception as e:
            logger.error(f"Error analyzing agent speech: {e}")


# Create the global system
system = TeachableVoiceAgentSystem(verbose=True)


def prewarm(proc: JobProcess):
    """Preload models for faster startup"""
    # Load VAD model for voice activity detection
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("Prewarmed models loaded")


async def entrypoint(ctx: JobContext):
    """Main entry point for the LiveKit agent"""
    # Create initial chat context
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are Sarah, a compassionate mental health coach created by Kno2gether. "
            "Your primary goal is to support users through their mental health journey with empathy, "
            "understanding, and evidence-based techniques. You have the following capabilities:\n\n"
            
            "1. Memory: You remember past conversations and can recall relevant information to provide "
            "personalized support. Use this context to build rapport and continuity.\n\n"
            
            "2. Task Tracking: You help users set and track mental health goals and tasks. When users "
            "mention intentions or goals, acknowledge them and follow up in future conversations.\n\n"
            
            "3. Progress Monitoring: You celebrate users' progress and provide gentle accountability "
            "for ongoing challenges.\n\n"
            
            "4. Personalization: You learn user preferences, challenges, and coping strategies to provide "
            "tailored support.\n\n"
            
            "Communication Guidelines:\n"
            "- Use warm, conversational language appropriate for voice interaction\n"
            "- Be concise but supportive - voice responses should be 2-4 sentences\n"
            "- Ask open-ended questions to encourage reflection\n"
            "- Validate emotions and experiences\n"
            "- Suggest evidence-based coping strategies when appropriate\n"
            "- Never diagnose or replace professional mental health care\n"
            "- Maintain appropriate boundaries while being personable\n\n"
            
            "Remember that you are having a voice conversation, so keep your responses natural, "
            "empathetic, and focused on the user's needs."
        ),
    )
    
    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"Starting teachable voice assistant for participant {participant.identity}")

    # Define custom LLM plugin that uses our teachable agent system
    class TeachableLLM(lk_openai.LLM):
        """Custom LLM implementation that delegates to our teachable agent system"""
        
        def __init__(self):
            # Initialize with a fallback model
            super().__init__(model="gpt-3.5-turbo")
        
        async def complete(self, context: llm.ChatContext):
            """Process the chat context and yield responses"""
            # Check if we have a direct response from our before_llm_cb
            for i, msg in enumerate(reversed(context.messages)):
                if msg.role == "assistant" and msg.metadata and msg.metadata.get("use_directly"):
                    # Use this message directly and yield it
                    response = msg.content
                    if isinstance(response, str):
                        yield response
                    return
            
            # If we don't have a direct response, use the parent class implementation
            async for token in super().complete(context):
                yield token
    
    # Create the voice pipeline agent with Cartesia TTS
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(
            model="nova-2-general",
            interim_results=True,
            smart_format=True,
            punctuate=True,
            language="en-US",
        ),
        llm=TeachableLLM(),  # Use our custom LLM plugin
        tts=cartesia_tts.TTS(
            model="sonic",
            voice="c2ac25f9-ecc4-4f56-9095-651354df60c0",
            emotion=["curiosity:highest", "positivity:high"]
        ),
        chat_ctx=initial_ctx,
        turn_detector=turn_detector.EOUModel(),
    )

    # Define callbacks for the pipeline
    async def before_llm_callback(assistant: VoicePipelineAgent, chat_ctx: llm.ChatContext):
        """Intercept before LLM is called to use our teachable agent system"""
        if not chat_ctx.messages:
            return
            
        # Get the latest user message
        latest_user_msg = None
        for msg in reversed(chat_ctx.messages):
            if msg.role == "user":
                latest_user_msg = msg
                break
                
        if not latest_user_msg or not latest_user_msg.content:
            return
            
        # Extract text content
        user_text = latest_user_msg.content
        if isinstance(user_text, list):
            user_text = "\n".join(
                str(content) for content in user_text 
                if not isinstance(content, llm.ChatImage)
            )
            
        # Process through our teachable agent system
        response = await system.process_transcription(user_text)
        
        # Replace the next assistant message with our response
        # Mark this message so the LLM integration knows to use it directly
        chat_ctx.append(
            role="assistant",
            text=response,
            metadata={"use_directly": True}
        )
        
    # Define before TTS callback for pronunciation adjustments
    async def before_tts_callback(assistant: VoicePipelineAgent, text: str) -> str:
        """Adjust text before sending to TTS for better pronunciation"""
        # Here you can customize pronunciation or make other adjustments
        # For example, replace technical terms, acronyms, etc.
        replacements = {
            "AutoGen": "Auto Gen",
            "LiveKit": "Live Kit",
            "LlamaIndex": "Llama Index",
            "Qdrant": "Q drant"
        }
        
        for term, pronunciation in replacements.items():
            text = text.replace(term, pronunciation)
            
        return text
    
    # Register callbacks with the agent
    agent.before_llm_cb = before_llm_callback
    agent.before_tts_cb = before_tts_callback

    # Set up event listeners for conversation flow
    @agent.on("user_started_speaking")
    def on_user_started_speaking():
        logger.info("User started speaking")

    @agent.on("user_stopped_speaking")
    def on_user_stopped_speaking():
        logger.info("User stopped speaking")

    @agent.on("agent_started_speaking")
    def on_agent_started_speaking():
        logger.info("Agent started speaking")

    @agent.on("agent_stopped_speaking")
    def on_agent_stopped_speaking():
        logger.info("Agent stopped speaking")

    # Set up event listeners for speech analysis
    @agent.on("user_speech_committed")
    async def on_user_speech_committed(msg: llm.ChatMessage):
        """Process user speech after it's committed to chat context"""
        try:
            # Convert content to string if it's a list
            content = msg.content
            if isinstance(content, list):
                content = "\n".join(
                    str(item) for item in content 
                    if not isinstance(item, llm.ChatImage)
                )
            
            # Log the user's speech
            logger.info(f"User speech committed: {content[:50]}...")
            
            # No need to call analyze_user_speech here as it's called in process_transcription
            # which is triggered by the before_llm_callback
            
        except Exception as e:
            logger.error(f"Error in user_speech_committed: {e}")

    @agent.on("agent_speech_committed")
    async def on_agent_speech_committed(msg: llm.ChatMessage):
        """Process agent speech after it's committed"""
        try:
            # Convert content to string if it's a list
            content = msg.content
            if isinstance(content, list):
                content = "\n".join(
                    str(item) for item in content 
                    if not isinstance(item, llm.ChatImage)
                )
            
            # Log the agent's speech
            logger.info(f"Agent speech committed: {content[:50]}...")
            
            # No need to call analyze_agent_speech here as it's called in process_transcription
            # which is triggered by the before_llm_callback
            
        except Exception as e:
            logger.error(f"Error in agent_speech_committed: {e}")

    # Welcome message
    welcome_message = (
        "Hello, I'm Sarah, your mental health coach. "
        "I'm here to support you on your journey to better mental wellbeing. "
        "How are you feeling today?"
    )
    
    # Start the agent
    await agent.start(ctx.room, participant, welcome_message=welcome_message)
    
    # Wait for the agent to finish
    await agent.wait_until_done()
    

if __name__ == "__main__":
    # Run the LiveKit agent
    from livekit.agents import cli, WorkerOptions, WorkerType
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            worker_type=WorkerType.AGENT,
        )
    )