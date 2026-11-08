from pathlib import Path
import os
from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Read the API key from Demo/.env
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

api_key = os.getenv("ANTHROPIC_API_KEY")
workspace_id = os.getenv("ANTHROPIC_WORKSPACE_ID")

# Client used for every Claude request
client = Anthropic(
    api_key=api_key,
    default_headers={"anthropic-workspace-id": workspace_id} if workspace_id else None,
)

app = FastAPI()

# Let the browser page on port 5500 call this API on port 8000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# GET /  — quick check that the server is running
@app.get("/")
def home():
    return {"message": "API is running"}


# POST /chat  — send the user's prompt to Claude and return the reply
@app.post("/chat")
def chat(body: dict):
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=500,
        messages=[{"role": "user", "content": body["prompt"]}],
    )
    return {"reply": response.content[0].text}
