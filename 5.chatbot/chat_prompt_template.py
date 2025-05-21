from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage

chat_template = ChatPromptTemplate([
    ('system',"You are a helpful {domain} expert"),
    ('human',"Explain in simplay way, whic is {topic}")
])

prompt = chat_template.invoke({"domain":"cricket","topic":"dusra"})

print(prompt)