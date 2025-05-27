from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

prompt = PromptTemplate(
    template = 'Generate 5 facts about {topic}',
    input_variables= ['topic']
)

model = ChatGoogleGenerativeAI(model ='gemini-2.0-flash')

parser = StrOutputParser()

chain = prompt | model | parser 

result = chain.invoke({'topic':'cricket'})

print(result)

chain.get_graph().print_ascii()