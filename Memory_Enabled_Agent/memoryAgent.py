#!/usr/bin/env python3

import os
import sys
import asyncio
import json
import logging
import uuid
import datetime
from typing import List, Dict, Optional, Any

# Load environment variables
from dotenv import load_dotenv

# AutoGen imports
from autogen import ConversableAgent, config_list_from_json
from autogen.agentchat.contrib.capabilities.teachability import Teachability

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
from livekit.plugins import deepgram, openai as lk_openai, silero, turn_detector
from livekit.plugins.cartesia import tts as cartesia_tts

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex, 
    Document, 
    StorageContext,
    load_index_from_storage
)
from llama_index.embeddings.openai import OpenAIEmbedding

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("teachable-voice-agent")

# Load environment variables
load_dotenv()

# Environment settings
OAI_CONFIG_LIST = os.environ.get("OAI_CONFIG_LIST", "oai_config_list.json")
KEY_LOC = os.environ.get("KEY_LOC", ".")

# Memory settings
MEMORY_PERSIST_DIR = os.environ.get("MEMORY_PERSIST_DIR", "./memory_storage")
INDEX_NAME = os.environ.get("INDEX_NAME", "teachable_agent_memory")

# LLM Model
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

class TeachableMemory:
    """Memory system for the teachable agent using LlamaIndex with local storage"""
    
    def __init__(self, persist_dir: str = MEMORY_PERSIST_DIR, verbose: bool = False):
        self.persist_dir = persist_dir
        self.verbose = verbose
        self.setup_vector_store()
    
    def setup_vector_store(self):
        """Setup local vector store for long-term memory"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.persist_dir, exist_ok=True)
            
            # Set up embedding model
            self.embed_model = OpenAIEmbedding()
            
            # Check if we have existing storage
            if os.path.exists(os.path.join(self.persist_dir, "docstore.json")):
                if self.verbose:
                    logger.info(f"Loading existing vector index from {self.persist_dir}")
                
                # Load existing index
                storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
                self.vector_index = load_index_from_storage(storage_context)
            else:
                if self.verbose:
                    logger.info("Creating new vector index")
                
                # Create new index
                self.vector_index = VectorStoreIndex(
                    [],
                    embed_model=self.embed_model
                )
                # Persist empty index
                self.vector_index.storage_context.persist(persist_dir=self.persist_dir)
            
            logger.info("Vector memory system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error setting up vector memory: {e}")
            raise
    
    async def store_interaction(self, user_input: str, agent_response: str):
        """Store the interaction in the vector database"""
        try:
            # Create a document from the interaction
            interaction_text = f"User: {user_input}\nAssistant: {agent_response}"
            interaction_doc = Document(
                text=interaction_text,
                metadata={
                    "timestamp": str(datetime.datetime.now()),
                    "type": "conversation"
                }
            )
            
            # Add to vector index
            self.vector_index.insert(interaction_doc)
            
            # Persist the updated index
            self.vector_index.storage_context.persist(persist_dir=self.persist_dir)
            
            if self.verbose:
                logger.info("Stored interaction in vector database")
                
        except Exception as e:
            logger.error(f"Error storing interaction: {e}")
    
    async def retrieve_relevant_context(self, query: str, limit: int = 3) -> str:
        """Retrieve relevant context from past interactions"""
        try:
            # Query vector store for relevant memories
            query_engine = self.vector_index.as_query_engine(similarity_top_k=limit)
            memory_results = query_engine.query(query)
            
            if self.verbose:
                logger.info(f"Retrieved {limit} relevant memories")
            
            return memory_results.response
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return ""


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
            )
            
            # Create data directory if it doesn't exist
            os.makedirs(f"./data/{self.agent_name}_db", exist_ok=True)
            
            # Add teachability capability
            teachability = Teachability(
                verbosity=self.verbose,
                reset_db=reset_db,
                path_to_db_dir=f"./data/{self.agent_name}_db",
                recall_threshold=1.5,
            )
            
            # Attach capability to agent
            teachability.add_to_agent(teachable_agent)
            
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


class TeachableVoiceAgentSystem:
    """Main system that combines the teachable agent with LiveKit voice capabilities"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.memory = TeachableMemory(verbose=verbose)
        self.teachable_agent = TeachableAgent(verbose=1)
        self.conversation_history = []
        
        if self.verbose:
            logger.info("TeachableVoiceAgentSystem initialized")
    
    async def process_transcription(self, text: str) -> str:
        """Process transcribed user input through the teachable agent with memory"""
        try:
            if self.verbose:
                logger.info(f"Processing: {text}")
            
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": text})
            
            # Retrieve relevant context from memory
            memory_context = await self.memory.retrieve_relevant_context(text)
            
            # Create enhanced prompt with memory context
            enhanced_prompt = text
            if memory_context:
                enhanced_prompt = f"{text}\n\nRelevant context from previous conversations: {memory_context}"
            
            # Generate response with teachable agent
            response = await self.teachable_agent.generate_response(enhanced_prompt)
            
            # Store in conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Store interaction in memory
            await self.memory.store_interaction(text, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing transcription: {e}")
            return "I'm having trouble processing that request."


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
            "You are a mental health specialist created by Kno2gether with teachable capabilities. "
            "You learn from user interactions and retain information shared with you. "
            "Try to apply your learned knowledge when responding to new queries. "
            "You should use clear, concise responses that work well for voice interaction."
        ),
    )

    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"Starting teachable voice assistant for participant {participant.identity}")

    # Define custom LLM plugin that uses our teachable agent
    class TeachableLLM(lk_openai.LLM):
        """Custom LLM implementation that delegates to our teachable agent system"""
        
        def __init__(self):
            # Initialize with a fallback model
            super().__init__(model="gpt-3.5-turbo")
        
        async def complete(self, context: llm.ChatContext) -> llm.AsyncChatIterator:
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
        """Intercept before LLM is called to use our teachable agent instead"""
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
    
    @agent.on("user_speech_committed")
    def on_user_speech_committed(msg: llm.ChatMessage):
        if isinstance(msg.content, list):
            content = "\n".join(
                "[image]" if isinstance(x, llm.ChatImage) else str(x) for x in msg.content
            )
        else:
            content = msg.content
        logger.info(f"User speech committed: {content}")
    
    @agent.on("agent_started_speaking")
    def on_agent_started_speaking():
        logger.info("Agent started speaking")
    
    @agent.on("agent_stopped_speaking")
    def on_agent_stopped_speaking():
        logger.info("Agent stopped speaking")
    
    @agent.on("agent_speech_committed")
    def on_agent_speech_committed(msg: llm.ChatMessage):
        logger.info(f"Agent speech committed: {msg.content}")
    
    @agent.on("agent_speech_interrupted")
    def on_agent_speech_interrupted():
        logger.info("Agent speech was interrupted")
    
    # Start the agent
    agent.start(ctx.room, participant)

    # Set up metrics collection
    usage_collector = metrics.UsageCollector()

    @agent.on("metrics_collected")
    def _on_metrics_collected(mtrcs: metrics.AgentMetrics):
        metrics.log_metrics(mtrcs)
        usage_collector.collect(mtrcs)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Welcome message
    await agent.say("Hello! I'm your Sarah and I m here to help. I can learn from our conversations. How can I help you today?", allow_interruptions=True)


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