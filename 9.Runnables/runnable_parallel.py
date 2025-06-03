from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

prompt1 = PromptTemplate(
    template = 'Generate a tweet about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate (
    template = 'Generate a LinkedIn post about {topic}',
    input_variables= ['topic']

)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet' : RunnableSequence( prompt1, model, parser),
    'LinkedIn' : RunnableSequence( prompt2, model, parser)
})

result = parallel_chain.invoke({'topic':'AI'})
print(result)