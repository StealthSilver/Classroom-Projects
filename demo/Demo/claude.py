from anthropic import Anthropic 

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
workspace_id = os.getenv("ANTHROPIC_WORKSPACE_ID")

client = Anthropic(api_key=api_key,
                  )
item = input("enter hte name of the product: ")
prompt = f"generate a 100 word description for hte {item}, in the description add the benifits and the feedbac for the{item}"
response = client.messages.create(
    model = "claude-sonnet-4-5",
    max_tokens = 400,
    messages = [
        {"role": "user", "content": prompt}
    ]
)

print(response.content[0].text)
