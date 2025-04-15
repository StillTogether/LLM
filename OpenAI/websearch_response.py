from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_KEY"])

response = client.responses.create(
    model="gpt-4o-mini",
    input="特朗普最新关税政策",
    tools=[{"type": "web_search_preview"}]
)
print(response.output_text)


