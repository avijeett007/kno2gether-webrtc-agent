import os
from dotenv import load_dotenv

def load_config():
    load_dotenv()
    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "livekit_api_key": os.getenv("LIVEKIT_API_KEY"),
        "livekit_api_secret": os.getenv("LIVEKIT_API_SECRET"),
        "tts_provider": os.getenv("TTS_PROVIDER", "openai"),
        "webhook_url": os.getenv("WEBHOOK_URL"),
        "gpt_model": os.getenv("GPT_MODEL", "gpt-4o-mini"),  # Default to gpt-4o-mini if not specified
        "openai_tts_voice": os.getenv("OPENAI_TTS_VOICE", "alloy"),  # Default to alloy if not specified
    }