from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnableParallel
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
import json

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

prompt1 = PromptTemplate(
    input_variables=['summary'],
    template="""\
# Evaluation
## Metric Definition
You are evaluating the **Conciseness** of the summary. A concise summary **does not contain unimportant information**. It includes only what is necessary and relevant to the task.

## Scoring Definition
**Conciseness Score (0–100):**  
Assign a score from 0 to 100 based on how well the summary avoids including unimportant details:
- **100** = No unimportant information at all.
- **80–99** = Mostly concise; contains very little non-essential content.
- **60–79** = Somewhat concise; several unnecessary parts exist.
- **0–59** = Not concise; too much irrelevant or unimportant content.

## Objective
Your task is to judge whether the summary includes only important, necessary information and avoids including anything that is not important. Focus strictly on whether the summary has removed all unimportant content.

## Evaluation Steps
- Look for any unimportant, irrelevant, or repeated content.
- Decide whether the summary could be shorter without losing any valuable meaning.
- The summary should still make sense after removing non-essential parts.
- Give a **conciseness score** and a **brief explanation** of why.

### Generated Summary:
"{summary}"

### Output Format:
{{"conciseness_score": <0–100>, "explanation": "<brief explanation of why this score was assigned>"}}
"""
)
# loader1= TextLoader('input1.txt',encoding='utf-8')
# input = loader1.load()

loader2= TextLoader('output1.txt',encoding='utf-8')
output = loader2.load()


parser = StrOutputParser()
chain = prompt1| model | parser 

result = chain.invoke({'summary':output[0].page_content})
# parsed_result = json.loads(result)
# print(parsed_result)
print(result)

