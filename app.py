from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from pydantic import BaseModel
import dspy
import os
import requests

# Load environment variables
load_dotenv()

app = FastAPI()

templates = Jinja2Templates(directory="templates")

class Agent:
    def ask(self, prompt):
        return f"ASK received: {prompt}"

    def plan(self, prompt):
        return f"PLAN received: {prompt}"

class CrossrefInput(BaseModel):
    prompt: str        

agent = Agent()


# ----------------------------
# Utility: Get Crossref Metadata
# ----------------------------
# Defining Tools
def crossref_search_json(query: str):
    try:
        url = f"https://api.crossref.org/works?query={query}&rows=5"
        r = requests.get(url, timeout=10)

        if r.status_code != 200:
            raise Exception(f"Crossref returned status {r.status_code}")

        data = r.json()
        items = data.get("message", {}).get("items", [])

        essentials = []
        for item in items:
            essentials.append({
                "doi": item.get("DOI"),
                "title": item.get("title", ["Untitled"])[0],
                "publisher": item.get("publisher"),
                "type": item.get("type"),
                "published": item.get("published-print", {})
                    .get("date-parts", [[None]])[0],
                "url": item.get("URL"),
            })

        return essentials

    except Exception as e:
        return {"error": str(e)}

# ----------------------------
# DSPy Agent Definition
# ----------------------------
class CrossrefService(dspy.Signature):
    """
    You are a Crossref agent that fetches the results from crossref api endpoint based on the given query.
    You can use the given tools to complete user requests.
    """
    query: str = dspy.InputField(desc="The search query to look up on Crossref.")
    process_result: str = dspy.OutputField(desc="Summarized result of the API response.")


# ----------------------------
# DSPy Configuration
# ----------------------------
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Missing GOOGLE_API_KEY or GEMINI_API_KEY in environment!")

# Configure Gemini as the LM backend for DSPy
dspy.configure(lm=dspy.LM("gemini/gemini-2.5-flash"))

# Create the DSPy ReAct agent
agent = dspy.ReAct(CrossrefService, tools=[crossref_search_json])
dspy_program = dspy.asyncify(agent)

# ----------------------------
# FastAPI Endpoint
# ----------------------------
@app.post("/crossref")
async def crossref(data: CrossrefInput):
    """Run the DSPy Crossref API Agent"""
    try:
        result = await dspy_program(query=data.prompt)
        return {"status": "success", "results": result.process_result}
    except Exception as e:
        print("DSPy ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, name="index.html")

@app.post("/ask")
async def ask(data: dict):
    return {"response": agent.ask(data["prompt"])}

@app.post("/plan")
async def plan(data: dict):
    return {"response": agent.plan(data["prompt"])}    