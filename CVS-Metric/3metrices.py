from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
import json

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

# Prompt 1 - Truthfulness
prompt1 = PromptTemplate(
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

# Prompt 2 - Completeness
prompt2 = PromptTemplate(
    input_variables=['note', 'summary'],
    template="""\
# Evaluation
## Metric Definition
You will be assessing summarization quality, which measures the overall ability to summarize text and include all the important information. The instructions for performing a summarization task and the context to be summarized are provided in the user prompt. The response should be shorter than the text in the context. The response should not contain information that is not present in the context.

## Criteria
Instruction following: The response demonstrates a clear understanding of the summarization task instructions, satisfying all of the instruction's requirements.
Groundedness: The response contains information included only in the context. The response does not reference any outside information.
Conciseness: The response summarizes the relevant details in the original text without a significant loss in key information without being too verbose or terse.

## Example
Prompt Template:
You are a medical summarization evaluator. Review whether the summary captures all important facts from the original note.

Instructions:
- Cover all relevant critical information from the note, and communication records for each drug.
- Include system events, actions taken, outcomes, and timestamps.
- Focus on dates, treatment changes, symptoms, drug name mentions, and provider instructions.
- Score Completeness from 0 to 100%.

Output format:
{{"completeness_score": <int from 0 to 100>, "missing_info": ["<key item 1>", "<key item 2>"]}}

Original Note:
"{note}"

Generated Summary:
"{summary}"
"""
)

# Prompt 3 - Conciseness
prompt3 = PromptTemplate(
    input_variables=['note', 'summary'],
    template="""\
# Evaluation
## Metric Definition
You will be assessing the Conciseness of the model's response, which measures its conciseness and ability to provide sufficient detail without being overly wordy or excessively brief.

## Criteria Definition
Verbosity: The response is appropriately concise, providing sufficient detail without using complex language to thoroughly address the prompt without being overly wordy or excessively brief.

## Example
Prompt Templates:
Evaluate the summary for conciseness. Does it include only essential information?

Instructions:
- Check if any parts are redundant or irrelevant.
- Score qualitatively and explain.
- Use tags: ["Concise", "Somewhat Verbose", "Verbose"]

Output format:
{{"conciseness_rating": "<Concise | Somewhat Verbose | Verbose>", "reasoning": "<explanation>"}}

Original Note:
"{note}"

Generated Summary:
"{summary}"
"""
)

# Load input and output documents
loader1 = TextLoader('input1.txt', encoding='utf-8')
loader2 = TextLoader('output1.txt', encoding='utf-8')

input_doc = loader1.load()[0].page_content
output_doc = loader2.load()[0].page_content

# Parser and parallel evaluation chain
parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'truthfulness': prompt1 | model | parser,
    'completeness': prompt2 | model | parser,
    'conciseness': prompt3 | model | parser
})

# Invoke the chain
raw_results = parallel_chain.invoke({'note': input_doc,'summary': output_doc})

# Parse string results into dictionaries
truthfulness_result = json.loads(raw_results['truthfulness'])
completeness_result = json.loads(raw_results['completeness'])
conciseness_result = json.loads(raw_results['conciseness'])

# Print the evaluation scores
print("\n=== Evaluation Results ===")
print(f"Truthfulness Score: {truthfulness_result['truthfulness_score']}")

print(f"Completeness Score: {completeness_result['completeness_score']}")

print(f"Conciseness Rating: {conciseness_result['conciseness_rating']}")
