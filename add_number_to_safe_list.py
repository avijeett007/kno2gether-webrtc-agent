# Download the helper library from https://www.twilio.com/docs/python/install
import os
from twilio.rest import Client

# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
account_sid = "****"
auth_token = "******"
client = Client(account_sid, auth_token)

safelist = client.verify.v2.safelist.create(phone_number="+*****")

print(safelist.sid)