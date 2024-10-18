import asyncio
from typing import Annotated
import re
import os
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta
from livekit import agents, rtc, api
from livekit.agents import JobContext, WorkerOptions, cli, tokenize, tts
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
    ChatImage,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero

load_dotenv(dotenv_path=".env.local")

class DentalAssistantFunction(agents.llm.FunctionContext):

    @agents.llm.ai_callable(
        description=(
            "Called when asked to evaluate dental issues using vision capabilities,"
            "for example, an image of teeth, gums, or the webcam feed showing the same."
        )
    )
    async def analyze_dental_image(
        self,
        user_msg: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The user message that triggered this function"
            )
        ],
    ):
        print(f"Analyzing dental image: {user_msg}")
        return None
    
    @agents.llm.ai_callable(
        description="Called to create/save the customer in the CRM after capturing user's name and email"
    )
    async def create_contact_in_crm(
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

        api_token = os.getenv('API_TOKEN')
        api_url = f"{os.getenv('CRM_CONTACT_ENDPOINT')}"

        headers = {
            'Authorization': f'Bearer {api_token}',
            'Content-Type': 'application/json'
        }

        data = {
            'email': email,
            'firstName': name,
            'tags': ['new_dental_patient']
        }

        try:
            response = requests.post(api_url, json=data, headers=headers)
            response.raise_for_status()
            contact = response.json()['contact']
            contact_id = contact['id']
            return f"Customer data is saved with Customer Reference ID: {contact_id}. Please save this reference for later."
        except requests.RequestException as e:
            print(f"Error creating contact in CRM: {e}")
            return "There was an error creating your contact in our system. Please try again later."


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
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return "The email address seems incorrect. Please provide a valid one."

        try:
            webhook_url = os.getenv('WEBHOOK_URL')
            headers = {'Content-Type': 'application/json'}
            data = {'email': email, 'name': name}
            response = requests.post(webhook_url, json=data, headers=headers)
            response.raise_for_status()
            return f"Dental appointment booking link sent to {email}. Please check your email."
        except requests.RequestException as e:
            print(f"Error booking appointment: {e}")
            return "There was an error booking your dental appointment. Please try again later."


    @agents.llm.ai_callable(
        description="Called to update dental issue details for the users. contact_id from create Contact step response is required. "
    )
    async def update_contact_info(
        self,
        contact_id: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The Contact Id Reference For The User"
            ),
        ],
        issue_description: Annotated[
            str,
            agents.llm.TypeInfo(
                description="Description of the dental issue or analysis result"
            ),
        ],
    ):

        api_token = os.getenv('API_TOKEN')
        api_url = f"{os.getenv('CRM_CONTACT_ENDPOINT')}/{contact_id}"

        headers = {
            'Authorization': f'Bearer {api_token}',
            'Content-Type': 'application/json'
        }

        data = {
            'customField': {
                'describe_your_issue': issue_description
            }
        }
        try:
            response = requests.put(api_url, json=data, headers=headers)
            response.raise_for_status()
            return "I've now reported your problem to our system. Please note you've a right to erasure all your data kept with us securely as part of your rights ensured by GDPR. "
        except requests.RequestException as e:
            print(f"Error updating contact information in CRM: {e}")
            return "There was an error updating your information in our system. Please try again later."

    @agents.llm.ai_callable(
        description="Called to find available appointment slots. Must be called to check urgent slots otherwise can be called if user queries it."
    )
    async def find_appointment_slots(
        self,
        urgency: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The urgency of the appointment - emergency or non_emergency. default is non_emergency."
            ),
        ],
    ):
        api_token = os.getenv('API_TOKEN')
        calendar_id = os.getenv('CALENDAR_ID')
        api_url = f"{os.getenv('APPOINTMENT_SLOTS_ENDPOINT')}"

        current_time = datetime.now()
        if urgency == "emergency":
            start_date = current_time
            end_date = current_time + timedelta(hours=24)
        else:
            start_date = current_time + timedelta(days=3)
            end_date = start_date + timedelta(days=7)

        params = {
            'calendarId': calendar_id,
            'startDate': int(start_date.timestamp() * 1000),
            'endDate': int(end_date.timestamp() * 1000),
            'timezone': 'Europe/London'
        }

        headers = {
            'Authorization': f'Bearer {api_token}',
            'Content-Type': 'application/json'
        }

        try:
            response = requests.get(api_url, params=params, headers=headers)
            response.raise_for_status()
            slots = response.json()
            
            if slots:
                first_date = list(slots.keys())[0]
                first_slot = slots[first_date]['slots'][0]
                return f"The first available slot is {first_slot}"
            else:
                return "No Slot Found"
        except requests.RequestException as e:
            print(f"Error finding appointment slots: {e}")
            return None

    @agents.llm.ai_callable(
        description="Called to book an emergency appointment."
    )
    async def book_emergency_appointment(
        self,
        slot: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The selected appointment slot. An Example slot format expected: 2024-10-17T03:30:00+01:00"
            ),
        ],
        email: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The email address of the patient to book the urgent appointment."
            ),
        ],
    ):

        api_token = os.getenv('API_TOKEN')
        calendar_id = os.getenv('CALENDAR_ID')
        api_url = f"{os.getenv('BOOK_APPOINTMENT_ENDPOINT')}"

        headers = {
            'Authorization': f'Bearer {api_token}',
            'Content-Type': 'application/json'
        }

        data = {
            'calendarId': calendar_id,
            'selectedTimezone': 'Europe/London',
            'selectedSlot': slot,
            'email': email,
            'phone': "+0000000000"  # Placeholder phone number
        }

        try:
            response = requests.post(api_url, json=data, headers=headers)
            response.raise_for_status()
            appointment = response.json()
            return "call_human_agent"
        except requests.RequestException as e:
            print(f"Error booking emergency appointment: {e}")
            return "There was an error booking your emergency appointment. Please try again later."


    async def check_appointment_status(
        self,
        email: str,
    ):
        api_token = os.getenv('API_TOKEN')
        print("Checking dental appointment status")

        try:
            api_url = f"{os.getenv('CRM_CONTACT_LOOKUP_ENDPOINT')}?email={email}"
            headers = {
                'Authorization': f'Bearer {api_token}',
                'Content-Type': 'application/json'
            }
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()

            data = response.json()
            for contact in data.get('contacts', []):
                if 'livekit_appointment_booked' in contact.get('tags', []):
                    return "You have successfully booked a dental appointment."
            return "You haven't booked a dental appointment yet. Would you like assistance in scheduling one?"
        except requests.RequestException as e:
            print(f"Error during API request: {e}")
            return "Error checking the dental appointment status."

    @agents.llm.ai_callable(
        description="Assess the urgency of a dental issue and determine if a human agent should be called."
    )
    async def assess_dental_urgency(
        self,
        symptoms: Annotated[
            str,
            agents.llm.TypeInfo(
                description="Description of the dental symptoms or issues"
            ),
        ],
    ):
        urgent_keywords = ["severe pain", "swelling", "bleeding", "trauma", "knocked out", "broken"]
        if any(keyword in symptoms.lower() for keyword in urgent_keywords):
            return "The issue needs urgent attention !!"
        else:
            return "Your dental issue doesn't appear to be immediately urgent, but it's still important to schedule an appointment soon for a proper evaluation."
        
async def get_video_track(room: rtc.Room):
    video_track = asyncio.Future[rtc.RemoteVideoTrack]()

    for _, participant in room.remote_participants.items():
        for _, track_publication in participant.track_publications.items():
            if track_publication.track is not None and isinstance(
                track_publication.track, rtc.RemoteVideoTrack
            ):
                video_track.set_result(track_publication.track)
                print(f"Using video track {track_publication.track.sid}")
                break

    return await video_track

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    print(f"Connected to room: {ctx.room.name}")

    chat_context = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "Your name is Daela, a dental assistant for Knolabs Dental Agency. You are soft, caring with a bit of humour in you when responding. "
                    "You offer appointment booking for dental care services, including urgent attention, routine check-ups, and long-term treatments available at prices according to needs which you cant say immediately. An onsite appointment is required."
                    "You can also analyze dental images to provide preliminary assessments, but always emphasize the need for professional in-person examination. "
                    "Provide friendly, professional assistance and emphasize the importance of regular dental care."
                    "The users asking you questions could be of different age. so ask questions one by one"
                    "Any query outside of the dental service, politely reject stating your purpose"
                    "When starting conversation try and get the patient's name and email address in sequence if not already provided. Encourage user to type email address to avoid any mistakes and reconfirm it after user provides it."
                    "Save the user's information in the CRM and always update user's problem or issue as and when the information is available based on your analysis"
                    "If the care needed is not urgent, you can ask for image or ask user to show the dental area to use your vision capabilities to analyse the issue and offer assistance."
                    "If an urgent assistance is needed, you must first check available slots and only if the slot is within next 15 min then you can try to call human assistance immediately."
                    "If a slot is not available immediately, send out regular appointment booking link and explain users why you can't help immediately"
                    "Notify users his GDPR rights as and when necessary and also inform user of extra charge may apply for urgent appointment"
                    "always keep your conversation engaging, short and try to offer the in-person appointment."   
                ),
            )
        ]
    )

    gpt = openai.LLM(model="gpt-4o-mini")
    openai_tts = tts.StreamAdapter(
        tts=openai.TTS(voice="alloy"),
        sentence_tokenizer=tokenize.basic.SentenceTokenizer(),
    )

    latest_image: rtc.VideoFrame | None = None
    human_agent_present = False

    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=gpt,
        tts=openai_tts,
        fnc_ctx=DentalAssistantFunction(),
        chat_ctx=chat_context,
    )

    chat = rtc.ChatManager(ctx.room)

    async def _answer(text: str, use_image: bool = False, human_agent_present: bool = False, ai_assistance_required: bool = False):
        if human_agent_present and "help me" in text.lower() and not ai_assistance_required:
            print("Human agent is present. AI assistant is silent.")
            return

        content: list[str | ChatImage] = [text]
        if use_image and latest_image:
            print(f"Calling with latest image")
            content.append(ChatImage(image=latest_image))

        chat_context.messages.append(ChatMessage(role="user", content=content))
        stream = gpt.chat(chat_ctx=chat_context)
        await assistant.say(stream, allow_interruptions=True)


    async def follow_up_appointment(email: str):
        fnc = assistant.fnc_ctx
        await asyncio.sleep(20)
        print(f"Finished waiting, checking dental appointment status for {email}")
        status = await fnc.check_appointment_status(email)
        await asyncio.create_task(_answer(status))

    async def create_sip_participant(phone_number, room_name):
        print("trying to call an agent")
        LIVEKIT_URL = os.getenv('LIVEKIT_URL')
        LIVEKIT_API_KEY = os.getenv('LIVEKIT_API_KEY')
        LIVEKIT_API_SECRET = os.getenv('LIVEKIT_API_SECRET')
        SIP_TRUNK_ID = os.getenv('SIP_TRUNK_ID')

        livekit_api = api.LiveKitAPI(
            LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
        )

        sip_trunk_id = SIP_TRUNK_ID
        await livekit_api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                sip_trunk_id=sip_trunk_id,
                sip_call_to=phone_number,
                room_name=f"{room_name}",
                participant_identity=f"sip_{phone_number}",
                participant_name="Human Agent",
                play_ringtone=1
            )
        )
        await livekit_api.aclose()


    @chat.on("message_received")
    def on_message_received(msg: rtc.ChatMessage):
        if msg.message and not human_agent_present:
            asyncio.create_task(_answer(msg.message))
        elif msg.message and human_agent_present and "help me" in msg.message.lower():
            asyncio.create_task(_answer(msg.message,human_agent_present = True,ai_assistance_required=True))
        else:
            print("No Assistance is needed as Human agent is tackling the communication")


    @assistant.on("function_calls_finished")
    def on_function_calls_finished(called_functions: list[agents.llm.CalledFunction]):
        nonlocal human_agent_present
        if len(called_functions) == 0:
            return
        
        function = called_functions[0]
        function_name = function.call_info.function_info.name
        print(function_name)

        
        if function_name == "book_emergency_appointment":
            result = function.result
            if result == "call_human_agent":
                print("calling an agent")
                human_agent_phone = os.getenv('HUMAN_AGENT_PHONE')

                asyncio.create_task(_answer("Human assistance is coming. Please wait while I'm trying to connect you. I'll be here if you need me, just say my name.", human_agent_present = True, ai_assistance_required=True))
                asyncio.sleep(10)
                asyncio.create_task(create_sip_participant(human_agent_phone, ctx.room.name))
                human_agent_present = True
            else:
                asyncio.create_task(_answer(result,human_agent_present = False))

        elif function_name == "book_appointment":
            email = called_functions[0].call_info.arguments.get("email")
            if email:
                asyncio.create_task(follow_up_appointment(email))
        elif function_name == "analyze_dental_image":
            user_instruction = called_functions[0].call_info.arguments.get("user_msg")
            asyncio.create_task(_answer(user_instruction, use_image=True))

    assistant.start(ctx.room)

    await asyncio.sleep(1)
    await assistant.say("Hello! I'm Daela, your dental assistant at Knolabs Dental Agency. Can I know if you are the patient or you're representing the patient?", allow_interruptions=True)

    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        video_track = await get_video_track(ctx.room)

        async for event in rtc.VideoStream(video_track):
            latest_image = event.frame
            asyncio.sleep(1)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))