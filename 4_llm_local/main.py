"""Small backend for Student GPT."""

import anthropic
from dotenv import load_dotenv
from fastapi import Body, FastAPI
from fastapi.responses import FileResponse

# Load ANTHROPIC_API_KEY from the local .env file.
load_dotenv()

app = FastAPI()


@app.get("/")
def home():
    return FileResponse("index.html")


@app.post("/chat")
def chat(prompt: str = Body(embed=True)):
    # Send the user's prompt to Claude and return its text reply.
    client = anthropic.Anthropic()
    answer = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}],
    )
    return {"response": answer.content[0].text}
