from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')
prompt1= PromptTemplate(
    template = 'write a joke about {topic}',
    input_variables= ['topic']
)
parser = StrOutputParser()

prompt2 = PromptTemplate(
    template= 'Explain the follwing joke -{text}',
    input_variables= ['text']
)

chain = RunnableSequence( prompt1 , model, parser, prompt2, model, parser)

result = chain.invoke({'topic':'AI'})
print(result)