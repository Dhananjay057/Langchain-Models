from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnableParallel,RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

prompt1 = PromptTemplate(
    template = 'Generate a joke about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template= 'Explain the follwing joke -{text}',
    input_variables= ['text']
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence( prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explanation': RunnableSequence( prompt2, model, parser)
})
final_chain = RunnableSequence(joke_gen_chain,parallel_chain)
result = final_chain.invoke({'topic':'AI'})
print(result)