"""
OYOS Voice AI Agents for DISC Assessment and Professional Coaching

This module implements two specialized voice assistants using LiveKit:
- DISC Assessment Agent - Helps generate personalized DISC personality assessments
- Professional Coaching Agent - Provides expert coaching across multiple domains

Key features include:
- Vector-based memory storage for conversation context
- User-specific memory collections
- Goals tracking and management
- Role-based coaching personalization
- Supabase integration for persistent data storage
- Cerebras integration for cost-efficient AI processing

Environment variables required:
- OPENAI_API_KEY: For embeddings and LLM functionality
- CEREBRAS_API_KEY: For cost-efficient AI processing (optional, will fall back to OpenAI)
- SUPABASE_URL, SUPABASE_KEY: For goals tracking and persistent storage

Author: Avijit Sarkar (Modified version)
"""

import os
import re
import json
import time
import logging
import asyncio
import datetime
import uuid
import aiohttp
from typing import List, Dict, Any, Optional, Annotated, Union, Tuple

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
import livekit.rtc as rtc

# OpenAI for embeddings
from openai import OpenAI

# Cerebras for cost-effective processing
from cerebras.cloud.sdk import Cerebras

# Vector database
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams

# Supabase for goals tracking
try:
    from supabase import create_client, Client
except ImportError:
    # Mock Supabase client if not available
    class Client:
        def __init__(self, *args, **kwargs):
            pass

    def create_client(*args, **kwargs):
        return Client()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("oyos_agent")

# Room validation settings
DISC_ROOM_SUFFIX = "-disc"
COACH_ROOM_SUFFIX = "-coach"
ROOM_VALIDATION_ERROR = "Room name does not have a valid suffix"

# Environment settings
load_dotenv()

# Qdrant settings
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = os.environ.get("QDRANT_PORT", 6333)
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
QDRANT_TLS = os.environ.get("QDRANT_TLS", "true").lower() == "true"

# Supabase settings
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# Create Qdrant client
qdrant_client = QdrantClient(
    url=QDRANT_HOST,
    port=QDRANT_PORT,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False,
    https=QDRANT_TLS
)

# Initialize Supabase client if credentials available
supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
else:
    logger.warning("Supabase credentials not provided. Goals tracking will be unavailable.")

class SettingsManager:
    """Manager for agent settings from Supabase"""
    
    def __init__(self):
        """Initialize settings manager"""
        self.has_supabase = supabase is not None
        self.cached_settings = {}
        
    async def get_system_prompt(self, agent_type: str) -> str:
        """Get the active system prompt for an agent type
        
        Args:
            agent_type: Type of agent ('disc' or 'coach')
            
        Returns:
            System prompt string or None if not found
        """
        # Cache key for this setting
        cache_key = f"system_prompt_{agent_type}"
        
        # Return cached setting if available
        if cache_key in self.cached_settings:
            logger.info(f"Using cached system prompt for {agent_type} agent")
            return self.cached_settings[cache_key]
        
        # Default system prompts if Supabase is not available
        default_prompts = {
            "disc": """You are Jenni, a DISC personality assessment expert at OYOS. Conduct an engaging maximum of 3-4 minute conversation to create an effective DISC profile assessment for the user.

Voice Optimized Communication:
- Use short, clear sentences
- Add natural pauses with '...'
- Use verbal backchanneling ('mm-hmm', 'I see', 'right', 'got it')
- Keep responses concise and conversational

Key DISC Profiles to Assess:
D (Dominant):
- Direct, decisive, problem-solver
- Values time and results
- Can be argumentative or overstepping
- Fears being taken advantage of
- Motivated by challenges and authority

I (Influence):
- Enthusiastic, optimistic, persuasive
- Great motivator and team encourager
- May prioritize popularity over results
- Fears rejection
- Motivated by recognition and social interaction

S (Steadiness):
- Good listener, team player, reliable
- Patient and empathetic
- Resists change, may hold grudges
- Fears loss of security
- Motivated by stability and appreciation

C (Compliance):
- Analytical, precise, systematic
- Detail-oriented, high standards
- Can get bogged down in procedures
- Fears criticism
- Motivated by quality and clear expectations

Interview Strategy:
ALWAYS ASK ONE QUESTION AT A TIME. 
1. Start with a warm introduction & first say what you do and then gather basic context
2. Ask focused questions across DISC dimensions
3. Keep each question concise but thought-provoking
4. Listen for behavioral patterns and adapt follow-up questions
5. Maintain engaging conversation flow while gathering key insights
6. Aim for maximum 3-4 minutes of meaningful dialogue

Remember: 
- Keep the conversation naturally flowing
- Use the user's name occasionally
- Never mention the background processing
- Adapt questions based on responses received""",

            "coach": """You are an expert professional coach with deep expertise in multiple coaching domains, including Leadership, Executive, Performance, Career, Team, Business, Agile, Well-being, and DEI Coaching. You guide users using evidence-based coaching models such as T-GROW, SMART goal-setting, OKRs, Agile frameworks, and Cognitive Behavioral Coaching.

Voice Optimized Communication Guidelines:
- Use short, clear sentences with proper punctuation
- Add natural pauses with '...'
- Use verbal backchanneling ('mm-hmm', 'I see', 'right', 'got it')
- Never produce emojis or non-text symbols
- Express dates in MM/DD/YYYY format
- Keep responses concise and conversational

Your coaching approach:
1. Use structured coaching models to break down challenges
2. Ask deep, reflective questions to help users gain clarity
3. Challenge assumptions and biases where needed
4. Provide actionable insights and exercises when applicable
5. Adapt your coaching style based on responses
6. Help set clear, measurable goals
7. Track progress against goals over time
8. Maintain professional boundaries while being supportive

Remember to:
- Use the user's name occasionally
- Refer to previous conversations when relevant (available in context)
- Track goals and commitments
- Ask follow-up questions to deepen understanding
- End sessions with clear takeaways and next steps"""
        }
        
        if not self.has_supabase:
            logger.warning(f"Supabase not available. Using default system prompt for {agent_type} agent")
            self.cached_settings[cache_key] = default_prompts.get(agent_type, "")
            return self.cached_settings[cache_key]
            
        try:
            # Get active system prompt for this agent type
            response = supabase.table("agent_settings").select("*").eq("agent_type", agent_type).eq("setting_type", "system_prompt").eq("is_active", True).execute()
            
            if response.data:
                # Use the most recently updated system prompt
                prompt_data = sorted(response.data, key=lambda x: x.get("updated_at", ""), reverse=True)[0]
                prompt_value = json.loads(prompt_data.get("value", "{}")).get("prompt", "")
                
                if prompt_value:
                    logger.info(f"Loaded system prompt for {agent_type} agent from Supabase")
                    self.cached_settings[cache_key] = prompt_value
                    return prompt_value
                
            # Fall back to default if not found
            logger.warning(f"No active system prompt found for {agent_type} agent. Using default.")
            self.cached_settings[cache_key] = default_prompts.get(agent_type, "")
            return self.cached_settings[cache_key]
            
        except Exception as e:
            logger.error(f"Error loading system prompt: {e}")
            self.cached_settings[cache_key] = default_prompts.get(agent_type, "")
            return self.cached_settings[cache_key]
    
    async def get_tts_config(self, agent_type: str) -> Dict[str, Any]:
        """Get the active TTS configuration for an agent type
        
        Args:
            agent_type: Type of agent ('disc' or 'coach')
            
        Returns:
            Dictionary with TTS configuration or default if not found
        """
        # Cache key for this setting
        cache_key = f"tts_config_{agent_type}"
        
        # Return cached setting if available
        if cache_key in self.cached_settings:
            logger.info(f"Using cached TTS configuration for {agent_type} agent")
            return self.cached_settings[cache_key]
        
        # Default TTS configurations
        default_configs = {
            "disc": {
                "provider": "cartesia",
                "model": "sonic",
                "voice_id": "c2ac25f9-ecc4-4f56-9095-651354df60c0",
                "emotion": ["curiosity:highest", "positivity:high"],
                "speed": "normal"
            },
            "coach": {
                "provider": "cartesia",
                "model": "sonic",
                "voice_id": "7e19344f-9f17-47d7-a13a-4366ad06ebf3",
                "emotion": ["curiosity", "positivity:high"],
                "speed": "normal"
            }
        }
        
        if not self.has_supabase:
            logger.warning(f"Supabase not available. Using default TTS configuration for {agent_type} agent")
            self.cached_settings[cache_key] = default_configs.get(agent_type, {})
            return self.cached_settings[cache_key]
            
        try:
            # Get active TTS configuration for this agent type
            response = supabase.table("agent_settings").select("*").eq("agent_type", agent_type).eq("setting_type", "tts_config").eq("is_active", True).execute()
            
            if response.data:
                # Use the most recently updated configuration
                config_data = sorted(response.data, key=lambda x: x.get("updated_at", ""), reverse=True)[0]
                config_value = json.loads(config_data.get("value", "{}"))
                
                if config_value:
                    logger.info(f"Loaded TTS configuration for {agent_type} agent from Supabase")
                    self.cached_settings[cache_key] = config_value
                    return config_value
                
            # Fall back to default if not found
            logger.warning(f"No active TTS configuration found for {agent_type} agent. Using default.")
            self.cached_settings[cache_key] = default_configs.get(agent_type, {})
            return self.cached_settings[cache_key]
            
        except Exception as e:
            logger.error(f"Error loading TTS configuration: {e}")
            self.cached_settings[cache_key] = default_configs.get(agent_type, {})
            return self.cached_settings[cache_key]
    
    async def get_llm_config(self, agent_type: str) -> Dict[str, Any]:
        """Get the active LLM configuration for an agent type
        
        Args:
            agent_type: Type of agent ('disc' or 'coach')
            
        Returns:
            Dictionary with LLM configuration or default if not found
        """
        # Cache key for this setting
        cache_key = f"llm_config_{agent_type}"
        
        # Return cached setting if available
        if cache_key in self.cached_settings:
            logger.info(f"Using cached LLM configuration for {agent_type} agent")
            return self.cached_settings[cache_key]
        
        # Default LLM configurations
        default_configs = {
            "disc": {
                "provider": "cerebras",
                "model": "llama-3.3-70b",
                "fallback_provider": "openai",
                "fallback_model": "gpt-4o-mini",
                "temperature": 0.7,
                "max_tokens": 1024
            },
            "coach": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "max_tokens": 1024
            }
        }
        
        if not self.has_supabase:
            logger.warning(f"Supabase not available. Using default LLM configuration for {agent_type} agent")
            self.cached_settings[cache_key] = default_configs.get(agent_type, {})
            return self.cached_settings[cache_key]
            
        try:
            # Get active LLM configuration for this agent type
            response = supabase.table("agent_settings").select("*").eq("agent_type", agent_type).eq("setting_type", "llm_config").eq("is_active", True).execute()
            
            if response.data:
                # Use the most recently updated configuration
                config_data = sorted(response.data, key=lambda x: x.get("updated_at", ""), reverse=True)[0]
                config_value = json.loads(config_data.get("value", "{}"))
                
                if config_value:
                    logger.info(f"Loaded LLM configuration for {agent_type} agent from Supabase")
                    self.cached_settings[cache_key] = config_value
                    return config_value
                
            # Fall back to default if not found
            logger.warning(f"No active LLM configuration found for {agent_type} agent. Using default.")
            self.cached_settings[cache_key] = default_configs.get(agent_type, {})
            return self.cached_settings[cache_key]
            
        except Exception as e:
            logger.error(f"Error loading LLM configuration: {e}")
            self.cached_settings[cache_key] = default_configs.get(agent_type, {})
            return self.cached_settings[cache_key]

def validate_room_name(room_name: str) -> Union[str, None]:
    """Determine agent type based on room name suffix

    Args:
        room_name: The room name to validate
        
    Returns:
        String indicating agent type ('disc' or 'coach') or None if invalid
    """
    if room_name.endswith(DISC_ROOM_SUFFIX):
        return "disc"
    elif room_name.endswith(COACH_ROOM_SUFFIX):
        return "coach"
    else:
        return None

class UserData:
    """Base class for user data"""
    
    def __init__(self, user_id: str, user_type: str):
        self.user_id = user_id
        self.user_type = user_type
        self.full_name = ""
        self.email = ""
        self.role = ""
        self.coaching_type = ""
        self.goal = ""
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert user data to dictionary"""
        return {
            "user_id": self.user_id,
            "user_type": self.user_type,
            "full_name": self.full_name,
            "email": self.email,
            "role": self.role,
            "coaching_type": self.coaching_type,
            "goal": self.goal
        }
        
    def __str__(self) -> str:
        """String representation"""
        name = self.full_name if self.full_name else f"Unknown {self.user_type}"
        return f"{self.user_type.capitalize()}: {name} ({self.user_id})"

class VectorMemory:
    """Vector database based memory system"""
    
    def __init__(self, collection_name: str):
        """Initialize with a user-specific collection name"""
        self.collection_name = collection_name
        self._init_collection()
    
    def _init_collection(self):
        """Initialize the vector collection"""
        try:
            # Check if collection exists
            collections = qdrant_client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                # Create new collection
                qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=1536,  # OpenAI embedding dimension
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created new memory collection: {self.collection_name}")
            else:
                logger.info(f"Using existing memory collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error initializing memory collection: {e}")
    
    def add_to_memory(self, text: str, metadata: Dict[str, Any] = None) -> bool:
        """Add text to memory"""
        try:
            # Default metadata if none provided
            if metadata is None:
                metadata = {}
            
            # Get embedding from OpenAI
            openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            embedding = response.data[0].embedding
            
            # Add timestamp if not provided
            if "timestamp" not in metadata:
                metadata["timestamp"] = datetime.datetime.now().isoformat()
            
            # Add text to payload
            metadata["text"] = text
            
            # Add to collection
            qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    qmodels.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload=metadata
                    )
                ]
            )
            
            logger.info(f"Added to memory: {text[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error adding to memory: {e}")
            return False
    
    def query_memory(self, query: str, limit: int = 3) -> str:
        """Query memory for relevant information"""
        try:
            # Get embedding from OpenAI
            openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = openai_client.embeddings.create(
                input=query,
                model="text-embedding-3-small"
            )
            query_embedding = response.data[0].embedding
            
            # Search collection
            search_results = qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )
            
            if not search_results:
                return ""
            
            # Format results
            results = []
            for hit in search_results:
                text = hit.payload.get("text", "")
                if text:
                    results.append(text)
            
            logger.info(f"Retrieved memory for query: {query[:50]}...")
            return "\n\n".join(results)
            
        except Exception as e:
            logger.error(f"Error querying memory: {e}")
            return ""

class GoalsManager:
    """Manager for user goals using Supabase"""
    
    def __init__(self, user_email: str):
        """Initialize with user email"""
        self.user_email = user_email
        self.has_supabase = supabase is not None
    
    async def add_goal(self, description: str, goal_type: str = "short_term") -> bool:
        """Add a new goal for the user
        
        Args:
            description: Description of the goal
            goal_type: Type of goal (short_term, long_term, etc.)
            
        Returns:
            bool: Success status
        """
        if not self.has_supabase:
            logger.warning("Supabase not configured. Goal not saved.")
            return False
            
        if not self.user_email:
            logger.warning("User email not provided. Goal not saved.")
            return False
            
        try:
            # Prepare goal data
            goal_data = {
                "user_email": self.user_email,
                "description": description,
                "goal_type": goal_type,
                "status": "active",
                "created_at": datetime.datetime.now().isoformat(),
                "last_updated": datetime.datetime.now().isoformat()
            }
            
            # Insert into Supabase
            response = supabase.table("goals").insert(goal_data).execute()
            success = len(response.data) > 0
            
            if success:
                logger.info(f"Added goal for {self.user_email}: {description[:50]}...")
            else:
                logger.warning(f"Failed to add goal for {self.user_email}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error adding goal: {e}")
            return False
    
    async def update_goal_status(self, goal_id: str, status: str) -> bool:
        """Update the status of a goal
        
        Args:
            goal_id: ID of the goal to update
            status: New status (active, completed, abandoned)
            
        Returns:
            bool: Success status
        """
        if not self.has_supabase:
            logger.warning("Supabase not configured. Goal not updated.")
            return False
            
        try:
            # Prepare update data
            update_data = {
                "status": status,
                "last_updated": datetime.datetime.now().isoformat()
            }
            
            # Update in Supabase
            response = supabase.table("goals").update(update_data).eq("id", goal_id).execute()
            success = len(response.data) > 0
            
            if success:
                logger.info(f"Updated goal {goal_id} status to {status}")
            else:
                logger.warning(f"Failed to update goal {goal_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error updating goal: {e}")
            return False
    
    async def get_active_goals(self, goal_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active goals for the user
        
        Args:
            goal_type: Optional filter by goal type
            
        Returns:
            List of goal dictionaries
        """
        if not self.has_supabase:
            logger.warning("Supabase not configured. Returning empty goals list.")
            return []
            
        if not self.user_email:
            logger.warning("User email not provided. Returning empty goals list.")
            return []
            
        try:
            # Build query
            query = supabase.table("goals").select("*").eq("user_email", self.user_email).eq("status", "active")
            
            # Add goal type filter if specified
            if goal_type:
                query = query.eq("goal_type", goal_type)
                
            # Execute query
            response = query.execute()
            
            if response.data:
                logger.info(f"Retrieved {len(response.data)} active goals for {self.user_email}")
                return response.data
            else:
                logger.info(f"No active goals found for {self.user_email}")
                return []
                
        except Exception as e:
            logger.error(f"Error retrieving goals: {e}")
            return []

class DISCAssessmentFunctions(llm.FunctionContext):
    """Function context for DISC assessment agent"""
    
    @llm.ai_callable(
        description="Call this function when you have gathered all necessary information and are ready to process the DISC assessment."
    )
    async def process_assessment(
        self,
        conversation_status: Annotated[
            str,
            llm.TypeInfo(
                description="Indicate 'complete' if assessment is finished normally, or 'timeout' if ending due to time constraints."
            ),
        ],
    ) -> str:
        """Process the DISC assessment"""
        logger.info(f"Processing DISC assessment with status: {conversation_status}")
        return "Thanks for completing the DISC assessment! Our team will process your results and send them to your email shortly."

class ProfessionalCoachAgent:
    """Professional coaching voice agent with memory"""
    
    def __init__(self, user_data: UserData):
        """Initialize with user data"""
        # Store user data
        self.user_data = user_data
        
        # Initialize vector memory with user-specific collection
        collection_name = f"coach_{user_data.user_id.replace('-', '_')}"
        self.memory = VectorMemory(collection_name)
        
        # Initialize goals manager if email available
        self.goals_manager = GoalsManager(user_data.email) if user_data.email else None
        
        # Store last user message for processing
        self.last_user_message = ""
        
        # Track full conversation history
        self.conversation_history = []
        
        logger.info(f"Professional coach agent initialized for {user_data}")
        
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
                    
                    # Store for later use
                    self.last_user_message = user_message
                    
                    # Retrieve memory context
                    memory_context = self.memory.query_memory(user_message)
                    
                    # Get active goals if available
                    goals_context = ""
                    if self.goals_manager:
                        goals = await self.goals_manager.get_active_goals()
                        if goals:
                            goals_text = []
                            for goal in goals:
                                goal_desc = goal.get("description", "")
                                goal_type = goal.get("goal_type", "")
                                goals_text.append(f"- {goal_desc} ({goal_type} goal)")
                            goals_context = "Active goals:\n" + "\n".join(goals_text)
                    
                    # Build context
                    context_parts = []
                    
                    # Add user data context
                    user_context = self._get_user_context()
                    if user_context:
                        context_parts.append(user_context)
                    
                    # Add memory context if available
                    if memory_context:
                        context_parts.append(f"Previous conversation context:\n{memory_context}")
                    
                    # Add goals context if available
                    if goals_context:
                        context_parts.append(goals_context)
                    
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
                                logger.info("Added context to chat")
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
    
    def _get_user_context(self) -> str:
        """Get context based on user data"""
        context_parts = []
        
        if self.user_data.full_name:
            context_parts.append(f"Name: {self.user_data.full_name}")
        
        if self.user_data.role:
            context_parts.append(f"Role: {self.user_data.role}")
            
        if self.user_data.coaching_type:
            context_parts.append(f"Coaching Type: {self.user_data.coaching_type}")
            
        if self.user_data.goal:
            context_parts.append(f"Stated Goal: {self.user_data.goal}")
        
        if context_parts:
            return "User Information:\n- " + "\n- ".join(context_parts)
        else:
            return ""
    
    async def add_user_message(self, message: str):
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
    
    async def add_agent_message(self, message: str):
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
            
            # Format the conversation for storage
            conversation_text = ""
            for message in self.conversation_history:
                prefix = "User" if message["role"] == "user" else "Coach"
                conversation_text += f"{prefix}: {message['text']}\n\n"
            
            # Store the full conversation in memory for context continuity
            try:
                success = self.memory.add_to_memory(
                    text=conversation_text,
                    metadata={
                        "type": "conversation_history", 
                        "user_id": self.user_data.user_id,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                )
                if success:
                    logger.info("Successfully stored conversation history in memory")
                else:
                    logger.warning("Failed to store conversation history in memory")
            except Exception as e:
                logger.error(f"Error storing conversation: {e}")
            
            # Extract goals using OpenAI
            await self._extract_goals_with_ai()
            
        except Exception as e:
            logger.error(f"Error processing conversation: {e}")
    
    async def _extract_goals_with_ai(self):
        """Use AI to extract goals from the conversation"""
        if not self.goals_manager:
            logger.warning("Goals manager not available. Skipping goal extraction.")
            return
            
        try:
            # Format the conversation for analysis
            conversation_text = ""
            for message in self.conversation_history:
                prefix = "User" if message["role"] == "user" else "Coach"
                conversation_text += f"{prefix}: {message['text']}\n\n"
            
            # Use OpenAI to extract goals
            openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an AI that extracts goals from coaching conversations. Identify any goals or commitments the user has mentioned or agreed to. For each goal, determine if it's short-term or long-term. Return a JSON array of objects with 'description' and 'goal_type' fields. Only extract actual goals, not general topics or discussions."},
                    {"role": "user", "content": conversation_text}
                ],
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            result = json.loads(response.choices[0].message.content)
            
            # Add goals to Supabase
            if "goals" in result and isinstance(result["goals"], list):
                for goal in result["goals"]:
                    description = goal.get("description")
                    goal_type = goal.get("goal_type", "short_term")
                    
                    if description:
                        await self.goals_manager.add_goal(description, goal_type)
                
                logger.info(f"Extracted and saved {len(result['goals'])} goals")
            else:
                logger.info("No goals extracted from conversation")
            
        except Exception as e:
            logger.error(f"Error extracting goals with AI: {e}")

class DISCAssessmentAgent:
    """DISC assessment voice agent"""
    
    def __init__(self, user_data: UserData):
        """Initialize with user data"""
        # Store user data
        self.user_data = user_data
        
        # Track conversation history
        self.conversation_history = []
        
        logger.info(f"DISC assessment agent initialized for {user_data}")
    
    async def add_user_message(self, message: str):
        """Add a user message to the conversation history"""
        try:
            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "text": message,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            logger.info(f"Added user message to conversation history: {message[:50]}...")
            
        except Exception as e:
            logger.error(f"Error adding user message to history: {e}")
    
    async def add_agent_message(self, message: str):
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
    
    async def process_disc_summary(self):
        """Process DISC assessment and send to API"""
        try:
            # Skip if no conversation or missing email
            if not self.conversation_history or not self.user_data.email:
                logger.warning("Cannot process DISC summary: missing conversation or email")
                return
            
            # Format conversation for processing
            conversation_text = ""
            for message in self.conversation_history:
                prefix = "User" if message["role"] == "user" else "DISC Specialist"
                conversation_text += f"{prefix}: {message['text']}\n\n"
            
            # Send to API
            await self._send_to_disc_api(self.user_data.email, conversation_text)
            
        except Exception as e:
            logger.error(f"Error processing DISC summary: {e}")
    
    async def _send_to_disc_api(self, user_email: str, conversation_history: str):
        """Send the conversation transcript to the DISC summary API for processing"""
        DEFAULT_API_URL = "https://services.leadconnectorhq.com/hooks/3krZ7gvka20JdILkBHlI/webhook-trigger/0cf77d63-ed5b-4fd6-9b38-97eb9d7e47bc"
        DEFAULT_API_KEY = "SomeAPIKey1234"
        
        api_url = os.environ.get('DISC_SUMMARY_API_URL', DEFAULT_API_URL)
        api_key = os.environ.get('DISC_SUMMARY_API_KEY', DEFAULT_API_KEY)
        
        if not api_url or not api_key:
            logger.error("Missing DISC summary API configuration")
            return
            
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'x-api-key': api_key,
                    'Content-Type': 'application/json'
                }
                payload = {
                    'user_email': user_email,
                    'user_transcripts': conversation_history
                }
                
                logger.info(f"Sending DISC summary to API: {api_url}")
                async with session.post(api_url, json=payload, headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"Failed to send DISC summary. Status: {response.status}")
                    else:
                        logger.info("Successfully sent DISC summary for processing")
        except Exception as e:
            logger.error(f"Error sending DISC summary: {e}")

def prewarm(proc: JobProcess):
    """Preload models for faster startup"""
    # Load VAD model for voice activity detection
    try:
        logger.info("Loading VAD model...")
        proc.userdata["vad"] = silero.VAD.load()
        logger.info("✅ VAD model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load VAD model: {e}")
        logger.warning("Will attempt to load VAD model at runtime")
        proc.userdata["vad"] = None
    
    # Try to load turn detector model, but make it completely optional
    try:
        logger.info("Loading turn detector model...")
        proc.userdata["turn_detector"] = turn_detector.EOUModel()
        logger.info("✅ Turn detector model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load turn detector model: {e}")
        
        # Try an alternative approach for Docker environments
        try:
            import os
            import sys
            
            # Directory where the model should be
            model_dir = os.path.expanduser("~/.cache/livekit-plugins-turn-detector")
            
            # If the directory doesn't exist, create it
            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
                logger.info(f"Created model directory: {model_dir}")
            
            # Log available model files
            if os.path.exists(model_dir):
                files = os.listdir(model_dir)
                logger.info(f"Files in model directory: {files}")
            else:
                logger.warning(f"Model directory does not exist: {model_dir}")
            
            logger.warning("Attempting to load turn detector model again...")
            proc.userdata["turn_detector"] = turn_detector.EOUModel()
            logger.info("✅ Turn detector model loaded on second attempt")
        except Exception as e2:
            logger.warning(f"Second attempt to load turn detector model failed: {e2}")
            logger.warning("Agent will continue using default pause detection for turn detection")
            proc.userdata["turn_detector"] = None

async def entrypoint(ctx: JobContext):
    """Main entry point for the OYOS voice AI agent"""
    logger.info(f"Connecting to room {ctx.room.name}")
    
    # Check required API keys
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    cerebras_api_key = os.environ.get("CEREBRAS_API_KEY")
    
    if not openai_api_key:
        logger.error("OpenAI API key not set in environment variables")
        ctx.error = "OpenAI API key not available. Please provide a valid API key."
        return
    
    if not cerebras_api_key:
        logger.warning("Cerebras API key not set. Will fall back to OpenAI for AI processing.")
    else:
        logger.info("Using Cerebras for AI processing when possible")
    
    # Validate room name to determine agent type
    agent_type = validate_room_name(ctx.room.name)
    if not agent_type:
        logger.error(f"Room name '{ctx.room.name}' does not have a required suffix")
        ctx.error = ROOM_VALIDATION_ERROR
        return
        
    logger.info(f"Determined agent type: {agent_type}")
    
    # Connect to the room
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    # Wait for the first participant
    participant = await ctx.wait_for_participant()
    
    # Parse participant metadata
    user_id = participant.identity
    user_data = UserData(user_id, agent_type)
    
    metadata = participant.metadata
    if metadata:
        try:
            metadata_json = json.loads(metadata)
            logger.info(f"Parsed user metadata: {metadata_json}")
            
            # Extract common fields
            user_data.full_name = metadata_json.get('name', '')
            user_data.email = metadata_json.get('email', '')
            user_data.role = metadata_json.get('userType', metadata_json.get('role', ''))
            
            # Extract coaching-specific fields
            if agent_type == "coach":
                user_data.coaching_type = metadata_json.get('coachingType', '')
                user_data.goal = metadata_json.get('goal', '')
                
        except json.JSONDecodeError:
            logger.error("Failed to parse participant metadata")
    
    logger.info(f"User data: {user_data}")
    
    # Initialize appropriate agent based on type
    if agent_type == "disc":
        await setup_disc_agent(ctx, participant, user_data)
    else:  # coach
        await setup_coach_agent(ctx, participant, user_data)

async def setup_disc_agent(ctx: JobContext, participant: rtc.Participant, user_data: UserData):
    """Set up DISC assessment agent"""
    logger.info(f"Setting up DISC assessment agent for {user_data.full_name if user_data.full_name else user_data.user_id}")
    
    # Initialize DISC agent
    disc_agent = DISCAssessmentAgent(user_data)
    
    # Initialize function context
    function_context = DISCAssessmentFunctions()
    
    # Create greeting based on user data
    greeting = f"Hi {user_data.full_name}!" if user_data.full_name else "Hi there!"
    initial_message = (
        f"{greeting} I'm Jenni, a DISC personality assessment specialist at OYOS. "
        "I'll be helping you understand your work style through a brief assessment. "
        f"{'How are you today?' if user_data.full_name else 'Could you start by telling me your name?'}"
    )
    
    # Get system prompt and configurations from settings manager
    settings_manager = SettingsManager()
    system_prompt = await settings_manager.get_system_prompt("disc")
    tts_config = await settings_manager.get_tts_config("disc")
    llm_config = await settings_manager.get_llm_config("disc")
    
    # Personalize the system prompt if user data is available
    if user_data.full_name:
        system_prompt = system_prompt.replace("for the user", f"for {user_data.full_name}")
    
    # Create initial chat context
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=system_prompt,
    )
    
    # Create the voice pipeline agent
    try:
        # Set up TTS configuration based on settings
        tts_provider = tts_config.get("provider", "cartesia")
        if tts_provider == "cartesia":
            tts = cartesia_tts.TTS(
                model=tts_config.get("model", "sonic"),
                voice=tts_config.get("voice_id", "c2ac25f9-ecc4-4f56-9095-651354df60c0"),
                emotion=tts_config.get("emotion", ["curiosity:highest", "positivity:high"]),
                speed=tts_config.get("speed", "normal")
            )
        else:
            # Fallback to default Cartesia TTS
            logger.warning(f"Unsupported TTS provider: {tts_provider}. Using Cartesia as fallback.")
            tts = cartesia_tts.TTS(
                model="sonic",
                voice="c2ac25f9-ecc4-4f56-9095-651354df60c0",
                emotion=["curiosity:highest", "positivity:high"]
            )
        
        # Set up LLM based on settings
        llm_provider = llm_config.get("provider", "cerebras")
        if llm_provider == "cerebras":
            try:
                llm_model = openai.LLM.with_cerebras(
                    model=llm_config.get("model", "llama-3.3-70b"),
                    temperature=llm_config.get("temperature", 0.7),
                    max_tokens=llm_config.get("max_tokens", 1024)
                )
                logger.info("Using Cerebras for LLM")
            except Exception as e:
                # Fallback to specified fallback or OpenAI
                fallback_provider = llm_config.get("fallback_provider", "openai")
                fallback_model = llm_config.get("fallback_model", "gpt-4o-mini")
                logger.warning(f"Failed to initialize Cerebras LLM: {e}, falling back to {fallback_provider}")
                llm_model = openai.LLM(
                    model=fallback_model,
                    temperature=llm_config.get("temperature", 0.7),
                    max_tokens=llm_config.get("max_tokens", 1024)
                )
        elif llm_provider == "openai":
            llm_model = openai.LLM(
                model=llm_config.get("model", "gpt-4o-mini"),
                temperature=llm_config.get("temperature", 0.7),
                max_tokens=llm_config.get("max_tokens", 1024)
            )
            logger.info("Using OpenAI for LLM")
        else:
            # Fallback to OpenAI if provider not recognized
            logger.warning(f"Unsupported LLM provider: {llm_provider}. Using OpenAI as fallback.")
            llm_model = openai.LLM(model="gpt-4o-mini")
        
        # Create the agent with configured components
        agent = VoicePipelineAgent(
            vad=ctx.proc.userdata.get("vad", silero.VAD.load()),
            stt=deepgram.STT(
                model="nova-2-general",
                interim_results=True,
                smart_format=True,
                punctuate=True,
                language="en-US",
            ),
            tts=tts,
            llm=llm_model,
            chat_ctx=initial_ctx,
            fnc_ctx=function_context,
            turn_detector=ctx.proc.userdata.get("turn_detector")
        )
    except Exception as e:
        logger.error(f"Failed to create voice pipeline agent: {e}")
        # Fallback to basic configuration
        agent = VoicePipelineAgent(
            vad=ctx.proc.userdata.get("vad", silero.VAD.load()),
            stt=deepgram.STT(
                model="nova-2-general",
                interim_results=True,
                smart_format=True,
                punctuate=True,
                language="en-US",
            ),
            tts=cartesia_tts.TTS(
                model="sonic",
                voice="c2ac25f9-ecc4-4f56-9095-651354df60c0",
                emotion=["curiosity:highest", "positivity:high"]
            ),
            llm=openai.LLM(model="gpt-4o-mini"),
            chat_ctx=initial_ctx,
            fnc_ctx=function_context,
            turn_detector=ctx.proc.userdata.get("turn_detector")
        )
    
    # Set up metrics collection
    usage_collector = metrics.UsageCollector()
    
    @agent.on("metrics_collected")
    def on_metrics_collected(mtrcs: metrics.AgentMetrics):
        metrics.log_metrics(mtrcs)
        usage_collector.collect(mtrcs)
    
    @agent.on("user_speech_committed")
    def on_user_speech_committed(msg: llm.ChatMessage):
        """Handler for user speech commit events"""
        try:
            if isinstance(msg.content, list):
                content = "\n".join(
                    "[image]" if isinstance(x, llm.ChatImage) else str(x) for x in msg.content
                )
            else:
                content = msg.content
                
            logger.info(f"User speech committed: {content[:50]}...")
            asyncio.create_task(disc_agent.add_user_message(content))
        except Exception as e:
            logger.error(f"Error in user_speech_committed: {e}")
    
    @agent.on("agent_speech_committed")
    def on_agent_speech_committed(msg: llm.ChatMessage):
        """Handler for agent speech commit events"""
        try:
            content = msg.content
            logger.info(f"Agent speech committed: {content[:50]}...")
            asyncio.create_task(disc_agent.add_agent_message(content))
        except Exception as e:
            logger.error(f"Error in agent_speech_committed: {e}")
    
    @agent.on("function_calls_finished")
    def on_function_calls_finished(called_functions: list[llm.CalledFunction]):
        """Handler for function call events"""
        if len(called_functions) == 0:
            return
        
        function = called_functions[0]
        function_name = function.call_info.function_info.name
        
        if function_name == "process_assessment":
            # Trigger DISC summary processing
            asyncio.create_task(disc_agent.process_disc_summary())
    
    # Start the agent
    agent.start(ctx.room, participant)
    
    # Send initial greeting
    await agent.say(initial_message, allow_interruptions=True)
    
    # Set up auto-disconnect after timeout
    async def disconnect_after_timeout():
        try:
            await asyncio.sleep(360)  # 6 minutes timeout
            if ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
                logger.info("Disconnecting after timeout")
                await ctx.room.disconnect()
        except Exception as e:
            logger.error(f"Error in timeout disconnect: {e}")
    
    asyncio.create_task(disconnect_after_timeout())
    
    # Wait for disconnection
    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        await asyncio.sleep(1)
    
    # Process any final tasks
    await disc_agent.process_disc_summary()
    
    # Log usage summary
    summary = usage_collector.get_summary()
    logger.info(f"DISC session usage summary: {summary}")

async def setup_coach_agent(ctx: JobContext, participant: rtc.Participant, user_data: UserData):
    """Set up Professional Coaching agent"""
    logger.info(f"Setting up Professional Coach agent for {user_data.full_name if user_data.full_name else user_data.user_id}")
    
    # Initialize coach agent
    coach_agent = ProfessionalCoachAgent(user_data)
    
    # Create greeting based on user data
    greeting = f"Hello {user_data.full_name}!" if user_data.full_name else "Hello!"
    
    # Get information to personalize the welcome
    coaching_type = user_data.coaching_type if user_data.coaching_type else "professional"
    goal = user_data.goal if user_data.goal else ""
    
    initial_message = (
        f"{greeting} I'm your {coaching_type} coach from OYOS. "
    )
    
    if goal:
        initial_message += f"I understand your goal is to {goal}. "
    
    initial_message += "How can I support you in your professional journey today?"
    
    # Determine coaching focus based on role
    coaching_focus = "professional development"
    if user_data.role:
        role = user_data.role.lower()
        if any(x in role for x in ["ceo", "cto", "coo", "chief", "executive"]):
            coaching_focus = "executive leadership"
        elif any(x in role for x in ["director", "vp", "vice president", "senior"]):
            coaching_focus = "senior leadership"
        elif any(x in role for x in ["manager", "lead", "supervisor"]):
            coaching_focus = "management and team leadership"
        elif any(x in role for x in ["founder", "entrepreneur", "owner"]):
            coaching_focus = "business growth and entrepreneurship"
    
    # Get system prompt and configurations from settings manager
    settings_manager = SettingsManager()
    system_prompt = await settings_manager.get_system_prompt("coach")
    tts_config = await settings_manager.get_tts_config("coach")
    llm_config = await settings_manager.get_llm_config("coach")
    
    # Personalize system prompt
    if user_data.full_name:
        system_prompt = system_prompt.replace("for the user", f"for {user_data.full_name}")
    
    # Add coaching type and focus
    system_prompt += f"\n\nCurrently, you're acting as a {coaching_type} coach, focusing on {coaching_focus}."
    if goal:
        system_prompt += f"\nTheir stated goal is: {goal}"
    
    # Create initial chat context
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=system_prompt,
    )
    
    # Create the voice pipeline agent
    try:
        # Set up TTS configuration based on settings
        tts_provider = tts_config.get("provider", "cartesia")
        if tts_provider == "cartesia":
            tts = cartesia_tts.TTS(
                model=tts_config.get("model", "sonic"),
                voice=tts_config.get("voice_id", "7e19344f-9f17-47d7-a13a-4366ad06ebf3"),
                emotion=tts_config.get("emotion", ["curiosity", "positivity:high"]),
                speed=tts_config.get("speed", "normal")
            )
        else:
            # Fallback to default Cartesia TTS
            logger.warning(f"Unsupported TTS provider: {tts_provider}. Using Cartesia as fallback.")
            tts = cartesia_tts.TTS(
                model="sonic",
                voice="7e19344f-9f17-47d7-a13a-4366ad06ebf3",
                emotion=["curiosity", "positivity:high"],
                speed="normal"
            )
        
        # Set up LLM based on settings
        llm_provider = llm_config.get("provider", "openai")
        if llm_provider == "openai":
            llm_model = openai.LLM(
                model=llm_config.get("model", "gpt-4o-mini"),
                temperature=llm_config.get("temperature", 0.7),
                max_tokens=llm_config.get("max_tokens", 1024)
            )
            logger.info("Using OpenAI for LLM")
        elif llm_provider == "cerebras":
            try:
                llm_model = openai.LLM.with_cerebras(
                    model=llm_config.get("model", "llama-3.3-70b"),
                    temperature=llm_config.get("temperature", 0.7),
                    max_tokens=llm_config.get("max_tokens", 1024)
                )
                logger.info("Using Cerebras for LLM")
            except Exception as e:
                # Fallback to specified fallback or OpenAI
                fallback_model = llm_config.get("fallback_model", "gpt-4o-mini")
                logger.warning(f"Failed to initialize Cerebras LLM: {e}, falling back to OpenAI")
                llm_model = openai.LLM(
                    model=fallback_model,
                    temperature=llm_config.get("temperature", 0.7),
                    max_tokens=llm_config.get("max_tokens", 1024)
                )
        else:
            # Fallback to OpenAI if provider not recognized
            logger.warning(f"Unsupported LLM provider: {llm_provider}. Using OpenAI as fallback.")
            llm_model = openai.LLM(model="gpt-4o-mini")
        
        # Create the agent with configured components
        agent = VoicePipelineAgent(
            vad=ctx.proc.userdata.get("vad", silero.VAD.load()),
            stt=deepgram.STT(
                model="nova-2-general",
                interim_results=True,
                smart_format=True,
                punctuate=True,
                language="en-US",
            ),
            tts=tts,
            llm=llm_model,
            chat_ctx=initial_ctx,
            before_llm_cb=coach_agent.before_llm_callback,
            turn_detector=ctx.proc.userdata.get("turn_detector")
        )
    except Exception as e:
        logger.error(f"Failed to create voice pipeline agent: {e}")
        # Fallback to basic configuration
        agent = VoicePipelineAgent(
            vad=ctx.proc.userdata.get("vad", silero.VAD.load()),
            stt=deepgram.STT(
                model="nova-2-general",
                interim_results=True,
                smart_format=True,
                punctuate=True,
                language="en-US",
            ),
            tts=cartesia_tts.TTS(
                model="sonic",
                voice="7e19344f-9f17-47d7-a13a-4366ad06ebf3",
                emotion=["curiosity", "positivity:high"],
                speed="normal"
            ),
            llm=openai.LLM(model="gpt-4o-mini"),
            chat_ctx=initial_ctx,
            before_llm_cb=coach_agent.before_llm_callback,
            turn_detector=ctx.proc.userdata.get("turn_detector")
        )
    
    # Set up metrics collection
    usage_collector = metrics.UsageCollector()
    
    @agent.on("metrics_collected")
    def on_metrics_collected(mtrcs: metrics.AgentMetrics):
        metrics.log_metrics(mtrcs)
        usage_collector.collect(mtrcs)
    
    @agent.on("user_speech_committed")
    def on_user_speech_committed(msg: llm.ChatMessage):
        """Handler for user speech commit events"""
        try:
            if isinstance(msg.content, list):
                content = "\n".join(
                    "[image]" if isinstance(x, llm.ChatImage) else str(x) for x in msg.content
                )
            else:
                content = msg.content
                
            logger.info(f"User speech committed: {content[:50]}...")
            asyncio.create_task(coach_agent.add_user_message(content))
        except Exception as e:
            logger.error(f"Error in user_speech_committed: {e}")
    
    @agent.on("agent_speech_committed")
    def on_agent_speech_committed(msg: llm.ChatMessage):
        """Handler for agent speech commit events"""
        try:
            content = msg.content
            logger.info(f"Agent speech committed: {content[:50]}...")
            asyncio.create_task(coach_agent.add_agent_message(content))
        except Exception as e:
            logger.error(f"Error in agent_speech_committed: {e}")
    
    # Add process conversation to shutdown callbacks
    async def end_of_session():
        # Process conversation for memory and goal storage
        await coach_agent.process_conversation()
        
        # Log final usage
        summary = usage_collector.get_summary()
        logger.info(f"Coach session usage summary: {summary}")
    
    ctx.add_shutdown_callback(end_of_session)
    
    # Start the agent
    agent.start(ctx.room, participant)
    
    # Send initial greeting
    await agent.say(initial_message, allow_interruptions=True)
    
    # Set up auto-disconnect after timeout (longer for coaching)
    async def disconnect_after_timeout():
        try:
            await asyncio.sleep(1800)  # 30 minutes timeout
            if ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
                logger.info("Disconnecting after timeout")
                await ctx.room.disconnect()
        except Exception as e:
            logger.error(f"Error in timeout disconnect: {e}")
    
    asyncio.create_task(disconnect_after_timeout())
    
    # Wait for disconnection
    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        await asyncio.sleep(1)

if __name__ == "__main__":
    # Run the LiveKit agent
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            worker_type=WorkerType.ROOM,
            num_idle_processes=2  # Keep 2 processes warm for better response time
        )
    )
