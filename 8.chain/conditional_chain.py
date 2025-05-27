from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

class Feedback(BaseModel):
    sentiment :Literal['positive','negative'] = Field(description = 'give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1= PromptTemplate(
    template = 
        "Classify the sentiment of the following feedback into 'positive' or 'negative'.\n"
        "Respond only in the following JSON format:\n"
        '{{"sentiment": "positive"}} or {{"sentiment": "negative"}}\n\n'
        "Feedback: {feedback}",
    input_variables = ['feedback']
)

parser = StrOutputParser()

classifier_chain = prompt1| model | parser2

prompt2 = PromptTemplate(
    template = 'Write an aapropriate response to this positive feedback \n {feedback}',
    input_variables= ['feedback']
)

prompt3 = PromptTemplate(
    template = 'Write an aapropriate response to this negative feedback \n {feedback}',
    input_variables= ['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser ),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser ),
    RunnableLambda(lambda x:"could not find sentiments")
)

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback':'This is a terrible smartphone'}))

chain.get_graph().print_ascii()