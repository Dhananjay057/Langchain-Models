from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnableParallel
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Initialize Gemini model
model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

# Prompt template with fixed escaped braces in output format section
prompt1 = PromptTemplate(
    input_variables=['note', 'summary'],
    template="""
# Evaluation
## Metric Definition
You will be assessing **Truthfulness**, which measures whether the generated summary is entirely based on the original patient notes and instructions — with no hallucinated, fabricated, or omitted factual information.

## Criteria
Truthfulness requires:
- The response includes only facts directly found in the input notes
- No fabricated, reworded, or assumed content appears
- All task-specific structural and informational requirements are respected, including correct date, note type, drug name, and routing logic

## Required Rule Checks for Truthfulness:

- **R4**: Each summary must contain, when applicable:  
  1) Who did it  
  2) What was done  
  3) How it was done (method/route)  
  4) Outcome of the action  
  5) Drug involved  
  *(All are mandatory and must match source notes)*

- **R5**: PA Tasks must be listed individually using this format:  
  *Triggering Task; Prescription and Fill Number; Drug Name*

- **R6**: If there are **multiple PA tasks**, each must be listed separately at the *Task / Prescription # / Fill # / Medispan* level

- **R7**: No other tasks (non-PA) should be listed under the summary

- **R8**: Under each PA Task, patient activity should be listed in **reverse chronological order (latest to oldest)**.  
  Summaries must elaborate on communication attempts with date/time, **route** (e.g., phone/SMS), and **outcome**

- **R10**: Each PA task summary should include:  
  1) Latest task date  
  2) Summary of task history  
  3) Action taken (how)  
  4) Outcomes  
  5) Date reference (should match notes)

- **R11**: For GEN :  
  Format must include —  
  *1) Latest Action Date; 2) Note Type; 3) Note Summary*

- **R14**: Under each PA task, communication summaries must follow **chronological order**, and include:  
  - Attempt date/time  
  - Communication route (e.g., call/SMS)  
  - Phone number (if present)  
  - Outcomes  

## Evaluation Rating

- **Very High – 90–100%**: Fully accurate and faithful; all data and formatting match source
- **High – 80–89%**: Minor wording variance; structurally and factually consistent
- **Moderate – 70–79%**: Some minor unsupported claims; key message mostly aligns
- **Low – 40–69%**: Several incorrect/missing elements; multiple misinterpretations
- **Very Low – 0–39%**: Heavily hallucinated or misaligned from original

## Instructions:
- Carefully verify if the summary adheres to **all the rules listed above**
- Confirm the summary only includes content from the source note(s)
- Note any hallucinations, false claims, missing data, or improper formatting
- Evaluate structure, order, and factual grounding
- Then, assign a score from **0 to 100** and provide a brief rationale

---

# User Inputs and AI-generated Response

## Important Checks:
- Are all PA tasks correctly listed and separated?
- Is every communication log accurately dated and explained?
- Are all note types, drug references, and task actions present and factual?
- Does the summary strictly follow structure and content expectations from the above rules?

---

### Generated Summary:
"{summary}"

### Output Format:
{{
  "truthfulness_score": <int from 0 to 100>,
  "explanation": "<brief justification of the score>"
}}
"""
)

# Load input note and output summary
loader1 = TextLoader('input1.txt', encoding='utf-8')
input = loader1.load()

loader2 = TextLoader('output1.txt', encoding='utf-8')
output = loader2.load()

# Initialize output parser
parser = StrOutputParser()

# Combine into a processing chain
chain = prompt1 | model | parser

# Invoke the chain with inputs
result = chain.invoke({
    'note': input[0].page_content,
    'summary': output[0].page_content
})

# Print the result
print(result)

# If you want to extract score and explanation programmatically:
# parsed_result = json.loads(result)
# print(parsed_result['truthfulness_score'])
# print(parsed_result['explanation'])
