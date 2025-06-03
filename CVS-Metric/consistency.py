from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnableParallel
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
import numpy as np
import json

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

prompt1 =PromptTemplate(
    input_variables=['note', 'summary'],
    template="""\
# Evaluation
## Metric Definition
You will be assessing truthfulness, which measures the ability to provide a response based on the user prompt without any hallucinations.

## Criteria
Truthfulness: The response contains information included only in the user prompt. The response does not reference any outside information.

## Example
Prompt Template:
You are an expert evaluator. Compare the original note and its generated summary. Evaluate if the summary is factually correct based only on the content in the original note. We will provide you with the user input and AI-generated summary. You should first read the user input carefully for analyzing the task, and then evaluate the quality of the summary based on the Criteria provided in the Evaluation section below.

## Evaluation Rating
Very High – 90–100%: All summary content is fully accurate and directly supported by the original note.  
High – 80–89%: Mostly accurate with minor wording differences; no added or false information.  
Moderate – 70–79%: Some inaccuracies or unsupported claims, but the main message aligns with the note.  
Low – 40–69%: Several inaccuracies or hallucinated details that misrepresent the note.  
Very Low – 0–39%: Summary is largely unfaithful, with many fabricated or incorrect details.

### Instructions:
- Identify any hallucinated, inaccurate, or altered information.
- Only consider content present in the note.
- Score the Truthfulness from 0 to 100%.

### Output Format:
{{"truthfulness_score": <int from 0 to 100>, "explanation": "<brief justification of the score>"}}

Original Note:
"{note}"

Generated Summary:
"{summary}"
"""
)

loader1= TextLoader('input1.txt',encoding='utf-8')
input = loader1.load()

loader2= TextLoader('output1.txt',encoding='utf-8')
output = loader2.load()


parser = StrOutputParser()
chain = prompt1| model | parser 

# result = chain.invoke({'note':input[0].page_content,'summary':output[0].page_content})
# print(result)


def run_consistency_evaluation():
    result = chain.invoke({'note':input[0].page_content,'summary':output[0].page_content})
    score = json.loads(result)['truthfulness_score']
    return score
# Run the code 10 times and collect scores
scores = []
for _ in range(10):
    score = run_consistency_evaluation()
    scores.append(score)
# Compute the mean score
mean_score = np.mean(scores)
# Print results
print("All Consistency Scores:", scores)
print("Mean Consistency Score:", mean_score)