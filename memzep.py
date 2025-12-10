import os
from zep_cloud.client import Zep
from zep_cloud.types import Message
from dotenv import load_dotenv
import uuid

load_dotenv()

client = Zep(api_key=os.getenv("ZEP_API_KEY"))

# You can choose any user ID, but we recommend using your internal user ID
user_id = "demo-graph-745b"

new_user = client.user.add(
    user_id=user_id,
    email="alice.bob@example.com",
    first_name="Alice",
    last_name="Bob",
)

thread_id = uuid.uuid4().hex # A new thread identifier
client.thread.create(
    thread_id=thread_id,
    user_id=user_id,
)

messages = [
    Message(
        name="Alice Bob",
        role="user",
        content="Who was Donald Trump?",
    )
]
response = client.thread.add_messages(thread_id, messages=messages)
# Get memory for the thread
memory = client.thread.get_user_context(thread_id=thread_id)
# Access the context block (for use in prompts)
context_block = memory.context
print(context_block)