from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, annotated_handlers
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

# Schema
class Review(BaseModel):

    summary: str
    sentiment : str

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The hardware is great, but the software feels bloated.There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands.Hoping for a software update to fix this.""")

print(result)
print(result.summary)
print(result.sentiment)