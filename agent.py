import asyncio
from typing import Annotated
import re
import os
import requests
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, tokenize, tts
from livekit.agents.llm import (
    ChatContext,
    ChatImage,
    ChatMessage,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero

# Load environment variables from .env.local
load_dotenv(dotenv_path=".env.local")

class AssistantFunction(agents.llm.FunctionContext):
    """This class defines functions that the assistant will call."""

    @agents.llm.ai_callable(
        description=(
            "Called when asked or needed to evaluate something that would require vision capabilities,"
            "for example, an image, video, or the webcam feed."
        )
    )
    async def provideImage(
        self,
        user_msg: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The user message that triggered this function"
            )
        ],
    ):
        print(f"Message triggering vision capabilities: {user_msg}")
        return None

    @agents.llm.ai_callable(
        description="Called when a user wants to book an appointment. This function sends a booking link to the provided email address and name."
    )
    async def book_appointment(
        self,
        email: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The email address to send the booking link to"
            ),
        ],
        name: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The name of the person booking the appointment"
            ),
        ],
    ):
        # Validate email
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return "The email address seems incorrect. Please provide a valid one."

        # Webhook call to book the appointment
        try:
            webhook_url = f'{os.getenv('WEBHOOK_URL')}'
            headers = {'Content-Type': 'application/json'}
            data = {'email': email, 'name': name}
            response = requests.post(webhook_url, json=data, headers=headers)
            response.raise_for_status()

            # Return success message
            return f"Appointment booking link sent to {email}. Please check your email."

        except requests.RequestException as e:
            print(f"Error booking appointment: {e}")
            return "There was an error booking your appointment. Please try again later."

    async def check_appointment_status(
        self,
        email: str,
    ):
        """Check if a user has booked an appointment based on their email."""
        api_token = os.getenv('API_TOKEN')
        print("calling check function")

        try:
            api_url = f'{os.getenv('CRM_CONTACT_LOOKUP_ENDPOINT')}?email={email}'
            headers = {
                'Authorization': f'Bearer {api_token}',
                'Content-Type': 'application/json'
            }
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()

            data = response.json()
            # Check if the contact has the 'livekit_appointment_booked' tag
            for contact in data.get('contacts', []):
                if 'livekit_appointment_booked' in contact.get('tags', []):
                    return "The user has successfully booked the appointment."
            return "The user has not yet booked an appointment. Please offer him help"

        except requests.RequestException as e:
            print(f"Error during API request: {e}")
            return "Error checking the appointment status."


async def check_room_status(room_name: str) -> bool:
    """Check if the room exists and has participants."""
    try:
        response = requests.get(f"{os.getenv('LIVEKIT_SERVER_URL')}/rooms/{room_name}", 
                                headers={"Authorization": f"Bearer {os.getenv('LIVEKIT_API_KEY')}:{os.getenv('LIVEKIT_API_SECRET')}"})
        if response.status_code == 200:
            room_info = response.json()
            participants = room_info.get("num_participants", 0)
            return participants > 0
        return False
    except requests.RequestException as e:
        print(f"Error checking room status: {e}")
        return False  # Assume room is inactive on error

async def wait_for_room(room_name: str) -> bool:
    """Poll for the room to be created and become active."""
    max_retries = 12  # Max number of retries (every 5 seconds) for up to 1 minute
    retries = 0
    while retries < max_retries:
        room_exists = await check_room_status(room_name)
        if room_exists:
            print(f"Room {room_name} is active. Joining now...")
            return True
        print(f"Waiting for the room {room_name} to become active... ({retries+1}/{max_retries})")
        retries += 1
        await asyncio.sleep(5)  # Poll every 5 seconds
    return False

async def entrypoint(ctx: JobContext):
    room_name = ctx.room.name

    room_active = await wait_for_room(room_name)  # Wait for client to start the room
    if not room_active:
        print(f"Room {room_name} did not become active in time.")
        return

    await ctx.connect()  # Agent joins the room after detecting activity
    print(f"Connected to room: {room_name}")

    chat_context = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "Your name is Daela, a sales assistant for Knolabs AI Agency. "
                    "You offer appointment booking for AI/Automation services through voice interaction."
                ),
            )
        ]
    )

    gpt = openai.LLM(model="gpt-4o-mini")
    openai_tts = tts.StreamAdapter(
        tts=openai.TTS(voice="alloy"),
        sentence_tokenizer=tokenize.basic.SentenceTokenizer(),
    )

    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=gpt,
        tts=openai_tts,
        fnc_ctx=AssistantFunction(),
        chat_ctx=chat_context,
    )

    chat = rtc.ChatManager(ctx.room)

    async def _answer(text: str):
        chat_context.messages.append(ChatMessage(role="user", content=text))
        stream = gpt.chat(chat_ctx=chat_context)
        await assistant.say(stream, allow_interruptions=True)

    async def follow_up_appointment(email: str):
        """Automatically check the appointment status and inform the user."""
        fnc = assistant.fnc_ctx
        await asyncio.sleep(20)  # Delay for checking (in seconds)
        print(f"Finished waiting, checking status for {email}")
        status = await fnc.check_appointment_status(email)
        await asyncio.create_task(_answer(status))

    @chat.on("message_received")
    def on_message_received(msg: rtc.ChatMessage):
        if msg.message:
            asyncio.create_task(_answer(msg.message))

    @assistant.on("function_calls_finished")
    def on_function_calls_finished(called_functions: list[agents.llm.CalledFunction]):
        if len(called_functions) == 0:
            return

        user_msg = called_functions[0].call_info.arguments.get("user_msg")
        if user_msg:
            asyncio.create_task(_answer(user_msg, use_image=True))
        email = called_functions[0].call_info.arguments.get("email")
        if email:
            asyncio.create_task(follow_up_appointment(email))

    assistant.start(ctx.room)

    await asyncio.sleep(1)
    await assistant.say("Hi there! How can I help?", allow_interruptions=True)

    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        await asyncio.sleep(1)  # Keep the connection alive for incoming events


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))