# chatbot.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Use Gemini Flash
model = genai.GenerativeModel("gemini-2.0-flash")

# Conversation history (keep in memory)
chat_session = model.start_chat(history=[])

def get_response(user_input: str) -> str:
    response = chat_session.send_message(user_input)
    return response.text
