import os
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

api_key = os.getenv("SARVAM_API_KEY")

if not api_key:
    raise ValueError("SARVAM_API_KEY was not found.")

client = SarvamAI(
    api_subscription_key=api_key
)

response = client.chat.completions(
    model="sarvam-105b",
    messages=[
        {
            "role": "user",
            "content": "Explain machine learning in one sentence."
        }
    ],
)

print(response.choices[0].message.content)