from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline
# import os

# os.environ['HF_HOME'] = 'D:/huggingface_cache'

text_gen = pipeline(
    "text-generation", 
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=100,
    temperature=0.5
)

llm = HuggingFacePipeline(pipeline=text_gen)

model = ChatHuggingFace(llm=llm)

result = model.invoke("what is the capital of India?")
print(result.content)