from livekit.agents.voice_assistant import VoiceAssistant # Corrected import
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.plugins import silero, deepgram, openai
from tts import get_tts_engine
from .functions import AssistantFunction

async def create_voice_assistant(config, chat_context):
    gpt = openai.LLM(model=config["gpt_model"])
    tts_engine = await get_tts_engine(config)

    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=gpt,
        tts=tts_engine,
        fnc_ctx=AssistantFunction(config["webhook_url"]),
        chat_ctx=chat_context,
    )

    return assistant