# main.py
# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from .chatbot1 import get_response

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    reply = get_response(req.message)
    return ChatResponse(response=reply)

