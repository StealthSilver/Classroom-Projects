from pathlib import Path
import os
from anthropic import Anthropic
from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware


load_dotenv(Path(__file__).resolve().parents[2]/".env")

api_key = os.getenv("ANTHROPIC_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
     allow_methods=["*"],
     allow_headers = ["*"]
)


client = Anthropic(
    api_key=api_key
)


@app.get("/")
def home():
    return {"message": "Backend is running"}

@app.post("/chat")
def chat(body: dict):
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        messages=[{"role": "user", "content": body["prompt"]}],
    )
    return {"reply": response.content[0].text}