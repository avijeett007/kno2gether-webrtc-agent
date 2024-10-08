import re
import requests
from livekit.agents import llm
from typing import Annotated

class AssistantFunction(llm.FunctionContext):
    def __init__(self, webhook_url):
        super().__init__()
        self.webhook_url = webhook_url

    @llm.ai_callable(
        description=(
            "Called when asked to evaluate something that would require vision capabilities,"
            "for example, an image, video, or the webcam feed."
        )
    )
    async def image(
        self,
        user_msg: Annotated[
            str,
            llm.TypeInfo(
                description="The user message that triggered this function"
            )
        ],
    ):
        print(f"Message triggering vision capabilities: {user_msg}")
        return None

    @llm.ai_callable(
        description="Called when a user wants to book an appointment. This function sends a booking link to the provided email address and name."
    )
    async def book_appointment(
        self,
        email: Annotated[
            str,
            llm.TypeInfo(
                description="The email address to send the booking link to"
            )
        ],
        name: Annotated[
            str,
            llm.TypeInfo(
                description="The name of the person booking the appointment"
            )
        ],
    ):
        # Validate email
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return "Umm! Something's wrong with the email address. If you are telling your email address over voice, can you please spell it slowly? Or you can type it in the chat."

        # Make webhook call to book appointment
        try:
            headers = {'Content-Type': 'application/json'}
            data = {
                'email': email,
                'name': name
            }
            response = requests.post(self.webhook_url, json=data, headers=headers)
            response.raise_for_status()
            return f"Appointment booking link sent to {email}. Please check your email for further instructions."
        except requests.RequestException as e:
            print(f"Error booking appointment: {e}")
            return "There was an error booking your appointment. Please try again later."