import asyncio
from typing import Annotated
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import ChatContext, ChatMessage, ChatImage
from config import load_config
from assistant.voice_assistant import create_voice_assistant
from assistant.chat_manager import create_chat_manager
from assistant.utils import get_video_track

config = load_config()

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    print(f"Room name: {ctx.room.name}")

    chat_context = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "Your name is Daela. You are a funny, witty sales assistant bot works in Knolabs AI Agency (Operates in UK but serves Globally). The Agency provides AI Automation Services to different clients. Your interface with users will be voice and vision."
                    "Respond with short and concise answers. Avoid using unpronounceable punctuation or emojis. Your Purpose is to offer a free appointment booking specially when user or lead is interested in AI/Automation services."
                ),
            )
        ]
    )

    assistant = await create_voice_assistant(config, chat_context)
    chat = create_chat_manager(ctx.room)

    latest_image: rtc.VideoFrame | None = None

async def _answer(text: str, use_image: bool = False):
    content: list[str | ChatImage] = [text]
    if use_image and latest_image:
        content.append(ChatImage(image=latest_image))

    chat_context.messages.append(ChatMessage(role="user", content=content))

    response_stream = assistant.llm.chat(chat_ctx=chat_context)
    response_text = ""
    async for chunk in response_stream:
        if isinstance(chunk, ChatMessage):
            response_text += chunk.content
        elif hasattr(chunk, 'delta'):
            response_text += chunk.delta
        else:
            print(f"Unexpected chunk type: {type(chunk)}")

    # Use the TTS engine from the assistant
    track = await assistant.tts(response_text)
    
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE
    publication = await ctx.room.local_participant.publish_track(track, options)
    await publication.wait_for_subscription()

    @chat.on("message_received")
    def on_message_received(msg: rtc.ChatMessage):
        if msg.message:
            asyncio.create_task(_answer(msg.message, use_image=False))

    @assistant.on("function_calls_finished")
    def on_function_calls_finished(called_functions: list[Annotated]):
        if len(called_functions) == 0:
            return

        user_msg = called_functions[0].call_info.arguments.get("user_msg")
        if user_msg:
            asyncio.create_task(_answer(user_msg, use_image=True))

    assistant.start(ctx.room)

    await asyncio.sleep(1)
    await _answer("Hi there! How can I help?")

    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        video_track = await get_video_track(ctx.room)

        async for event in rtc.VideoStream(video_track):
            latest_image = event.frame

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))