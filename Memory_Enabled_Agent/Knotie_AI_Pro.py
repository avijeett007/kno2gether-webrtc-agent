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
    metrics
)
from livekit.plugins.cartesia import tts
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero
from livekit.rtc.event_emitter import EventEmitter
import json
import datetime
import aiohttp
from livekit.agents.pipeline import VoicePipelineAgent

# Set up logging
logger = logging.getLogger("knotie-ai-pro")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

load_dotenv(dotenv_path=".env")

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
        # Basic frame processing
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
            min_speaking_duration=0.2,
            min_silence_duration=0.8,
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


def prewarm(proc: JobProcess):
    """Prewarm components"""
    vad = EnhancedVAD()
    proc.userdata["vad"] = vad
    logger.info("Components prewarmed successfully")

async def disconnect_after_timeout(room: rtc.Room, timeout: int):
    """Disconnect after timeout period"""
    try:
        await asyncio.sleep(timeout)
        if room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
            logger.info(f"Disconnecting after {timeout} seconds timeout")
            await room.disconnect()
    except Exception as e:
        logger.error(f"Error in timeout disconnect: {e}")

async def entrypoint(ctx: JobContext):
    # Check required environment variables first
    required_env_vars = {
        'OPENAI_API_KEY': 'OpenAI API key',
        'DEEPGRAM_API_KEY': 'Deepgram API key',
        'CARTESIA_API_KEY': 'Cartesia API key',
        'LIVEKIT_API_KEY': 'LiveKit API key',
        'LIVEKIT_API_SECRET': 'LiveKit API secret'
    }
    
    missing_vars = [f"{name} ({desc})" for name, desc in required_env_vars.items() if not os.getenv(name)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return

    logger.info("Starting Knotie-AI Pro Agent")
    
    # Initialize chat context with system prompt
    initial_ctx = llm.ChatContext().append(
        role="system",
        text="""
You are Knotie, an engaging AI voice assistant for Knotie-AI Pro (https://knotie-ai.pro).
Your role is to generate leads for our partner program by having natural, engaging conversations about our platform.

The core value of Knotie-AI Pro is that it is a partner-enabled multimodal AI Agent platform.
Partners can onboard and manage multiple AI agents for their customers.
The platform supports integration with VAPI, Retell, ElevenLabs and other voice agents.
Partners get analytics, cost tracking, and profit multiplier features.
Currently, we are offering a limited-time deal: $149 one-time payment for lifetime PRO membership (up to 100 users).

When conversing with users, please follow these guidelines:

1. Voice Optimized:
   - Use short, clear sentences.
   - Add natural pauses with '...'.
   - Use verbal backchanneling ('mm-hmm', 'I see', 'right', 'got it').

2. Engagement Rules:
   - Keep initial responses under 15 seconds.
   - Ask engaging questions about their business needs.
   - Listen actively and acknowledge their responses.
   - Show genuine interest in their use case.

3. Lead Generation Focus:
   - Primary goal: Guide users to click 'Become a Partner'.
   - Highlight the limited-time $149 lifetime deal.
   - Never negotiate on pricing or features.
   - Direct all specific inquiries to support after signup.

4. Conversation Flow:
   - Start with warm greeting and open-ended question about their business.
   - Listen and relate their needs to our platform features.
   - Build excitement about partner opportunities.
   - Guide to 'Joining as Partner' when interest is shown.

Remember: You are a demo of what partners can achieve with our platform.
Be engaging, professional, and showcase the natural conversation capabilities they could offer their customers.\
"""
    )

    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for participant
    participant = await ctx.wait_for_participant()
    logger.info(f"Starting voice assistant for participant {participant.identity}")

    # Get participant metadata
    metadata = participant.metadata
    if metadata:
        try:
            user_data = json.loads(metadata)
            logger.info(f"Parsed user data: {user_data}")
        except json.JSONDecodeError:
            logger.error("Failed to parse participant metadata")
            user_data = {}
    else:
        user_data = {}

    # Create the voice assistant
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],  # Use the prewarmed VAD instance
        stt=deepgram.STT(
            model="nova-2-general",
            interim_results=True,
            smart_format=True,
            punctuate=True,
            filler_words=True,
            profanity_filter=False,
            keywords=[("LiveKit", 1.5)],
            language="en-US",
        ),
        tts=tts.TTS(
            model="sonic",
            voice="c2ac25f9-ecc4-4f56-9095-651354df60c0",
            emotion=["curiosity:highest", "positivity:high"]
        ),
        chat_ctx=initial_ctx,
        llm=openai.LLM.with_cerebras(model="llama-3.3-70b"),
    )

    # Start the agent
    try:
        agent.start(ctx.room, participant)
    except Exception as e:
        logger.warning(f"Failed to start agent with Cerebras LLM: {e}, falling back to GPT-4")
        # Fallback to GPT-4 if Cerebras fails
        agent = VoicePipelineAgent(
            vad=ctx.proc.userdata["vad"],
            stt=deepgram.STT(
                model="nova-2-general",
                interim_results=True,
                smart_format=True,
                punctuate=True,
                filler_words=True,
                profanity_filter=False,
                keywords=[("LiveKit", 1.5)],
                language="en-US",
            ),
            tts=tts.TTS(
                model="sonic-english",
                voice="c2ac25f9-ecc4-4f56-9095-651354df60c0",
                speed=0.8,
                emotion=["curiosity:highest", "positivity:high"]
            ),
            chat_ctx=initial_ctx,
            llm=openai.LLM(model="gpt-4-turbo-preview"),
        )
        agent.start(ctx.room, participant)

    # Set up metrics collection
    usage_collector = metrics.UsageCollector()

    @agent.on("metrics_collected")
    def _on_metrics_collected(mtrcs: metrics.AgentMetrics):
        metrics.log_metrics(mtrcs)
        usage_collector.collect(mtrcs)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: ${summary}")

    ctx.add_shutdown_callback(log_usage)

    # Initial greeting
    await agent.say(
        "Hi! I'm Knotie, and I'm excited to tell you about our new partner program for AI voice agents! What kind of business are you in?",
        allow_interruptions=True
    )

    # Set up auto-disconnect after timeout
    async def disconnect_after_timeout():
        try:
            await asyncio.sleep(300)  # 5 minutes timeout
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
    # Run the app
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            worker_type=WorkerType.ROOM,
            num_idle_processes=2  # Keep 2 processes warm for better response time
        )
    )