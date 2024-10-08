from livekit.agents import tts, tokenize
from livekit.plugins import openai

async def create_openai_tts(config):
    return tts.StreamAdapter(
        tts=openai.TTS(voice=config["openai_tts_voice"]),
        sentence_tokenizer=tokenize.basic.SentenceTokenizer(),
    )