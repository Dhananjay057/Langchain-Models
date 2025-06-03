from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnableLambda,RunnablePassthrough, RunnableBranch
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

prompt1= PromptTemplate(
    template = 'Write a detailed report on the {topic}',
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template = 'summarise the follwing text \n {text}',
    input_variables= ['text']
)

parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x:len(x.split())>500, RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain,branch_chain)

print(final_chain.invoke({'topic':'Russia vs Ukraine'}))