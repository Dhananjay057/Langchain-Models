from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'google/gemma-2-2b-it',
    task = 'text-generation'
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name ='fact1', description="fact1 about the topic"),
    ResponseSchema(name ='fact2', description="fact2 bout the topic"),
    ResponseSchema(name ='fact3', description="fact3 about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = "Give me 3 facts about the {topic}\n {format_instruction}",
    input_variables=['topic'],
    partial_variables= {'format_instruction': parser.get_format_instructions()}
)

# prompt = template.invoke({'topic':'black hole'})
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
chain = template | model | parser
final_result= chain.invoke({"topic":"black hole"})
print(final_result)

