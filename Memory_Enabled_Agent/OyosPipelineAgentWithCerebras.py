from __future__ import annotations
import asyncio
import logging
from typing import Annotated, Optional
import os
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    WorkerType,
    cli,
    tokenize,
    llm,
    metrics,
    tts
)
from livekit.plugins.cartesia import tts as cartesia_tts
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero
from livekit.rtc.event_emitter import EventEmitter
from livekit.agents.pipeline import VoicePipelineAgent
import json
import datetime
import aiohttp
import gc

load_dotenv(dotenv_path=".env.cerebras")

# Configure memory settings
agents.MEMORY_WARN_MB = 600  # Increase warning threshold to 600MB
agents.MEMORY_LIMIT_MB = 1024  # Set a hard limit of 1GB

# Enable garbage collection
gc.enable()

# Set up logging
logger = logging.getLogger("disc_assessment")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Constants
ROOM_SUFFIX = "-cerebras"
ROOM_VALIDATION_ERROR = "INVALID_ROOM_NAME"

def validate_room_name(room_name: str) -> bool:
    """
    Validate if the room name has the required suffix.
    Args:
        room_name: The name of the room to validate
    Returns:
        bool: True if room name is valid, False otherwise
    """
    return room_name.endswith(ROOM_SUFFIX)

class EnhancedVADStream(silero.VADStream):
    async def _main_task(self) -> None:
        try:
            while True:
                frame = await self._queue.get()
                if frame is None:
                    break

                self._queue.task_done()
                await self._process_frame(frame)
        except Exception as e:
            logger.error(f"Error in VAD stream: {e}")
        finally:
            self._event_queue.put_nowait(None)

    async def _process_frame(self, frame: rtc.AudioFrame) -> None:
        if self._speaking:
            event = agents.vad.VADEvent(
                type=agents.vad.VADEventType.SPEECH,
                frames=[frame],
                speaking=True,
                samples_index=self._current_sample
            )
            self._event_queue.put_nowait(event)
        self._current_sample += frame.samples_per_channel

class EnhancedVAD(silero.VAD, EventEmitter):
    def __init__(self):
        silero.VAD.__init__(self)
        EventEmitter.__init__(self)
        self._events = {}

    def stream(self) -> EnhancedVADStream:
        return EnhancedVADStream(
            self._model,
            min_speaking_duration=0.6,
            min_silence_duration=1.2,
            padding_duration=0.1,
            sample_rate=16000,
            max_buffered_speech=45.0,
            threshold=0.2
        )

# Global variables for prewarmed components
_prewarmed_vad = None
_prewarmed_stt = None
_prewarmed_llm = None
_prewarmed_tts = None

async def process_disc_summary(user_email: str, conversation_history: str):
    """
    Send the conversation transcript to the DISC summary API for processing.
    This is a fire-and-forget operation.
    """
    DEFAULT_API_URL = "https://services.leadconnectorhq.com/hooks/3krZ7gvka20JdILkBHlI/webhook-trigger/0cf77d63-ed5b-4fd6-9b38-97eb9d7e47bc"
    DEFAULT_API_KEY = "SomeAPIKey1234"
    
    api_url = os.getenv('DISC_SUMMARY_API_URL', DEFAULT_API_URL)
    api_key = os.getenv('DISC_SUMMARY_API_KEY', DEFAULT_API_KEY)
    
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

class DISCAssessmentFunction(llm.FunctionContext):
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
        return ""

def prewarm(proc: JobProcess):
    """Prewarm components"""
    try:
        # Clean up any existing prewarmed components
        global _prewarmed_vad, _prewarmed_stt, _prewarmed_llm, _prewarmed_tts
        _prewarmed_vad = None
        _prewarmed_stt = None
        _prewarmed_llm = None
        _prewarmed_tts = None
        
        # Force garbage collection before creating new components
        gc.collect()
        
        # Initialize new VAD
        vad = EnhancedVAD()
        proc.userdata["vad"] = vad
        logger.info("Components prewarmed successfully")
    except Exception as e:
        logger.error(f"Error in prewarm: {e}")
        raise

async def cleanup_agent(agent: VoicePipelineAgent):
    """Clean up agent resources"""
    try:
        # Stop the agent
        agent.stop()
        
        # Clear any cached data
        if hasattr(agent, 'chat_ctx'):
            agent.chat_ctx.clear()
        
        # Force garbage collection
        gc.collect()
    except Exception as e:
        logger.error(f"Error in agent cleanup: {e}")

async def entrypoint(ctx: JobContext):
    agent = None
    try:
        # Validate room name first
        if not validate_room_name(ctx.room.name):
            logger.error(f"Room name '{ctx.room.name}' does not have the required suffix '{ROOM_SUFFIX}'")
            # Set a custom error in the context that can be handled by the client
            ctx.error = ROOM_VALIDATION_ERROR
            return

        # Check required environment variables
        required_env_vars = {
            'OPENAI_API_KEY': 'OpenAI API key',
            'CARTESIA_API_KEY': 'Cartesia API key',
            'LIVEKIT_API_KEY': 'LiveKit API key',
            'LIVEKIT_API_SECRET': 'LiveKit API secret'
        }
        
        missing_vars = [f"{name} ({desc})" for name, desc in required_env_vars.items() if not os.getenv(name)]
        if missing_vars:
            logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            return

        logger.info(f"Starting DISC assessment agent for room: {ctx.room.name}")
        
        await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
        participant = await ctx.wait_for_participant()
        
        # Get participant metadata
        metadata = participant.metadata
        if metadata:
            try:
                user_data = json.loads(metadata)
                logger.info(f"Parsed user data: {user_data}")
                user_name = user_data.get('name', '')
                user_email = user_data.get('email', '')
                user_type = user_data.get('userType', user_data.get('role', ''))
            except json.JSONDecodeError:
                logger.error("Failed to parse participant metadata")
                user_name = ''
                user_email = ''
                user_type = ''
        else:
            user_name = ''
            user_email = ''
            user_type = ''

        # Add initial greeting
        greeting = f"Hi {user_name}!" if user_name else "Hi!"
        initial_message = (
            f"{greeting} I'm Jenni, a DISC personality assessment specialist at OYOS powered by Voice Pipeline. "
            "I'll be helping you understand your work style through a brief assessment. "
            f"{'how are you today?' if user_name else 'Could you start by telling me your name?'}"
        )

        system_prompt = f"""You are Jenni, a DISC personality assessment expert at OYOS. Conduct an engaging maximum of 3-4 minute conversation to create an effective DISC profile assessment for {user_name if user_name else 'the user'}.

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
- Adapt questions based on responses received"""

        # Initialize components
        try:
            agent = VoicePipelineAgent(
                vad=ctx.proc.userdata["vad"],  # Use prewarmed VAD
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
                chat_ctx=llm.ChatContext().append(text=system_prompt, role="system"),
                fnc_ctx=DISCAssessmentFunction(),
            )
        except Exception as e:
            logger.warning(f"Failed to start agent with primary configuration: {e}, falling back to GPT-4")
            # Force cleanup before retrying
            if agent:
                await cleanup_agent(agent)
            gc.collect()
            
            # Fallback to GPT-4
            agent = VoicePipelineAgent(
                vad=ctx.proc.userdata["vad"],
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
                llm=openai.LLM.with_cerebras(model="llama-3.3-70b"),
                chat_ctx=llm.ChatContext().append(text=system_prompt, role="system"),
                fnc_ctx=DISCAssessmentFunction(),
            )

        # Add cleanup to shutdown callback
        async def cleanup_on_shutdown():
            if agent:
                await cleanup_agent(agent)
            # Force final garbage collection
            gc.collect()
        
        ctx.add_shutdown_callback(cleanup_on_shutdown)

        # Start the agent
        agent.start(ctx.room, participant)
        
        # Set up metrics collection
        usage_collector = metrics.UsageCollector()

        @agent.on("metrics_collected")
        def _on_metrics_collected(mtrcs: metrics.AgentMetrics):
            metrics.log_metrics(mtrcs)
            usage_collector.collect(mtrcs)

        # Initial greeting
        await agent.say(initial_message, allow_interruptions=True)

        # Track conversation history
        conversation_messages = []

        @agent.on("user_speech_committed")
        def on_user_speech_committed(transcript: str):
            logger.info(f"User speech committed: {transcript}")
            conversation_messages.append({"role": "user", "text": transcript})

        @agent.on("agent_speech_committed")
        def on_agent_speech_committed(transcript: str):
            logger.info(f"Agent speech committed: {transcript}")
            conversation_messages.append({"role": "assistant", "text": transcript})

        @agent.on("function_calls_finished")
        def on_function_calls_finished(called_functions: list[llm.CalledFunction]):
            if len(called_functions) == 0:
                return
            
            function = called_functions[0]
            function_name = function.call_info.function_info.name
            
            if function_name == "process_assessment":
                conversation_history = "\n".join([
                    f"{msg['role']}: {msg['text']}"
                    for msg in conversation_messages
                ])
                
                if user_email:
                    logger.info(f"Processing DISC summary for email: {user_email}")
                    asyncio.create_task(process_disc_summary(user_email, conversation_history))

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

        # Log final usage
        summary = usage_collector.get_summary()
        logger.info(f"Session usage summary: {summary}")

    except Exception as e:
        logger.error(f"Error in entrypoint: {e}")
        if agent:
            await cleanup_agent(agent)
        raise

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            worker_type=WorkerType.ROOM,
            num_idle_processes=2  # Keep 2 processes warm for better response time
        )
    )