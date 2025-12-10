from google import genai
from mem0 import Memory
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

config = {
    "embedder": {
        "provider": "gemini",
        "config": {"model": "models/text-embedding-004"}
    },
    "llm": {
        "provider": "gemini",
        "config": {
            "model": "gemini-1.5-flash",   # safer for free tier
            "temperature": 0.0,
            "max_tokens": 2000,
        },
    },
    "vector_store": {"config": {"embedding_model_dims": 768}}
}

memory = Memory.from_config(config)

system_prompt = "You are a helpful AI. Answer based on the query and stored memories."

def to_mem0_messages(history):
    """Convert Gemini-style history → Mem0 format"""
    msgs = []
    for item in history:
        role = item["role"]
        text = item["parts"][0]["text"]
        if role == "user":
            msgs.append({"role": "user", "content": text})
        else:
            msgs.append({"role": "assistant", "content": text})
    return msgs

def chat_with_memories(history, user_id="default_user"):
    query = history[-1]["parts"][0]["text"]

    # retrieve memories
    relevant = memory.search(query=query, user_id=user_id, limit=5)
    memories_str = "\n".join(f"- {m['memory']}" for m in relevant["results"])

    # build system instruction
    full_prompt = f"{system_prompt}\n\nUser Memories:\n{memories_str}"

    # generate response
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=history,
        system_instruction=full_prompt
    )

    # update chat history
    history.append({
        "role": "model",
        "parts": [{"text": response.text}]
    })

    # add memory
    mem0_messages = to_mem0_messages(history)
    memory.add(mem0_messages, user_id=user_id)

    return history

def main():
    print("Chat with Gemini (type 'exit' to quit)")
    history = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        history.append({"role": "user", "parts": [{"text": user_input}]})
        updated_history = chat_with_memories(history)
        print("Gemini:", updated_history[-1]["parts"][0]["text"])

main()
