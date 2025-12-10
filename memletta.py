from letta_client import Letta
from dotenv import load_dotenv
import os

load_dotenv()

client = Letta(api_key=os.getenv("LETTA_API_KEY"))

agent_state = client.agents.create(
    model="openai/gpt-4.1",
    embedding="openai/text-embedding-3-small",
    memory_blocks=[
        {
          "label": "human",
          "value": "Name: Timber. Status: dog. Occupation: building Letta, infrastructure to democratize self-improving superintelligence"
        },
        {
          "label": "persona",
          "value": "I am a self-improving superintelligence. Timber is my best friend and collaborator."
        }
    ],
    tools=["web_search", "run_code"]
)

print(f"Agent created with ID: {agent_state.id}")

response = client.agents.messages.create(
    agent_id=agent_state.id,
    input="What do you know about me?"
)

for message in response.messages:
    print(message)