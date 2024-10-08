from .openai_tts import create_openai_tts

async def get_tts_engine(config):
    if config["tts_provider"] == "openai":
        return await create_openai_tts(config)
    else:
        raise ValueError(f"Unsupported TTS provider: {config['tts_provider']}")