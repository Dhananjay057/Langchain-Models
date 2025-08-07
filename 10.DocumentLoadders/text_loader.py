from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

prompt = PromptTemplate(
    template = 'Write a summary for the following {poem}',
    input_variables= ['poem']
)

loader = TextLoader('cricket.txt',encoding='utf-8')
docs=loader.load()
 
parser = StrOutputParser()
# print(docs[0])
# print(docs[0].page_content)

chain = prompt | model | parser

print(chain.invoke({'poem': docs[0].page_content}))