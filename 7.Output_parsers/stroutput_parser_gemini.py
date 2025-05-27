from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
# from langchain_core.output_parsers import StrOutputParser

load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

# Prompt for full report
template = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# Prompt for summary
template1 = PromptTemplate(
    template='Write a 5 line summary on the following text. \n {text}',
    input_variables=['text']
)

# Create prompt from template
prompt = template.invoke({'topic':'black hole'})
result = model.invoke(prompt)

# Create summary prompt from result
summary_prompt = template1.invoke({'text':result.content})
summary = model.invoke(summary_prompt)
print(summary.content)
