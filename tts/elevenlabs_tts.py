from elevenlabs import generate, stream
from livekit import rtc
import asyncio

async def create_elevenlabs_tts(api_key, voice_id):
    async def tts_wrapper(text: str):
        audio_stream = generate(
            text=text,
            voice=voice_id,
            model="eleven_monolingual_v1",
            stream=True
        )

        source = rtc.AudioSource(sample_rate=44100, num_channels=1)
        track = rtc.LocalAudioTrack.create_audio_track("agent-mic", source)

        for chunk in audio_stream:
            await source.capture_frame(rtc.AudioFrame(chunk, 44100, 1))

        return track

    return tts_wrapper