from anthropic import Anthropic
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
workspace_id = os.getenv("ANTHROPIC_WORKSPACE_ID")
if not api_key:
    raise SystemExit("ANTHROPIC_API_KEY is missing. Add it to the .env file.")

client = Anthropic(
    api_key=api_key,
    default_headers={"anthropic-workspace-id": workspace_id} if workspace_id else None,
)
prompt = input("Enter a prompt: ")
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=500,
    messages=[{"role": "user", "content": prompt}],
)

print(response.content[0].text)