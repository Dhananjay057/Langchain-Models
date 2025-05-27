from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'google/gemma-2-2b-it',
    task = 'text-generation'
)

model = ChatHuggingFace(llm=llm)

# prompt1
template = PromptTemplate(
    template = 'Write a detailed report on {topic}',
    input_variables = ['topic']
)

#prompt2
template1 = PromptTemplate(
    template = 'write a 5 line summary on the following text./n {text}',
    input_variables = ["text"]
)

prompt = template.invoke({'topic':'blackhole'})
result = model.invoke(prompt)

prompt1 = template1.invoke({'text': result.content})
result1 = model.invoke(prompt1)
print(result1.content)
