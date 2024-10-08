from livekit import rtc

def create_chat_manager(room: rtc.Room):
    return rtc.ChatManager(room)