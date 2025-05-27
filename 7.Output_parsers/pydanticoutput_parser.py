from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'google/gemma-2-2b-it',
    task = 'text-generation'
)

model = ChatHuggingFace(llm=llm)     

class Person(BaseModel):

    name: str = Field(description="name of the person")
    age : int = Field(gt=18 , description="age of the person")
    city: str = Field(description="Name of the city the person resides")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template = "Give me the Name, age and city of {place} person \n {format_instruction}",
    input_variables=['place'],
    partial_variables= {"format_instruction":parser.get_format_instructions}
)

# prompt = template.invoke({"place":"indian"})
# result= model.invoke(prompt)
# final_result=parser.parse(result.content)

chain = template | model | parser
final_result = chain.invoke({"place":"indian"})
print(final_result)
