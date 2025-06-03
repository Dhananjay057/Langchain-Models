from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnableParallel
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
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

You are an expert evaluator. Compare the original note and its generated summary. Evaluate if the summary is factually correct based only on the content in the original note. We will provide you with the user input and AI-generated summary. You should first read the user input carefully for analyzing the task, and then evaluate the quality of the summary based on the Criteria provided in the Evaluation section below

## Evaluation Rating

Very High – 90-100% : All summary content is fully accurate and directly supported by the original note.
High – 80-89% : Mostly accurate with minor wording differences; no added or false information.
Moderate – 70-79% : Some inaccuracies or unsupported claims, but the main message aligns with the note.
Low – 40-69% : Several inaccuracies or hallucinated details that misrepresent the note.
Very Low – 0-39% : Summary is largely unfaithful, with many fabricated or incorrect details.

Instructions:
- Identify any hallucinated, inaccurate, or altered information.
- Only consider content present in the note.
- Score the Truthfulness from 0 to 100%.


Output format:{{ "truthfulness_score": <int from 0 to 100>, "explanation": "<brief justification of the score>"}}


# User Inputs and AI-generated Response
## User Inputs
"{note}"

### Prompt given for summary generation
(## **1. Gen notes summary:**
 
#### **Objective:**
- Write the summary from the **patientNoteList** with **noteTypeInd**: **GENERAL** or **noteTypeCd**: **GEN**  only in 100 words.
 
#### **Instructions**:
- Review the provided **patientNoteList** for **noteTypeInd**: **GENERAL** or **noteTypeCd**: **GEN** only.
- From column **noteSummaryTxt** and **noteTxt**, summarize each activity detailing what happen, how it happen, and how each issue addressed or resolved.
- Identify which activity is related to which medicineName or rxNumber and assign the respective activities to the genNotesSummary column of their respective Drug name or rxNumber where exact drugName matches in the json schema.
- There are some summary where exact medicineName is not present, identify such summary and assign the summary to every genNotesSummary of the schema where first word of the drugName matches with the medicineName present in the summary.
#### **Constraints:**
- If multiple summary are on the same date then just combine all those summary.
- Identify the drug name and extract its first word 'FirstWord'. If a summary is present under 'noteSummaryTxt' or 'noteTxt' which only contains the 'FirstWord' of the drug name, then the summary should be mentioned in the summaries of all the drugs with that 'FirstWord'. You MUST only match the 'FirstWord' of the drug name, if it does not match DO NOT list the summary under that drug.
- DO NOT summarize the notes of one drug into another drug summary. CHECK if the drug name present in the notes is the same as the drug you are placing the summary in.

#### **Recap:**
- If multiple summaries are on the same date then just combine it.


### **2. Bill notes summary:**
 
#### **Objective:**
- Write the summary from the **patientNoteList** with **noteTypeInd**: ***BILLING/PRICING/PRIOR AUTH** or **noteTypeCd**: **BIL**
 
#### **Instructions:**
- Review the provided **patientNoteList** for **noteTypeInd: BILLING/PRICING/PRIOR AUTH** or **noteTypeCd**: **BIL**.
- From column **noteSummaryTxt** and **noteTxt**, summarize each activity detailing what happen, how it happen, and how each issue addressed or resolved.
- Based on value in Created date and Account number, identify which activity is related to which Drug name or rxNumber and assign the respective activities to the bilNotesSummary column of their respective Drug name or task name in the json schema.
- Identify the drug name and extract its first word 'FirstWord'. If a summary is present under 'noteSummaryTxt' or 'noteTxt' which only contains the 'FirstWord' of the drug name, then the summary should be mentioned in the summaries of all the drugs with that 'FirstWord'. You MUST only match the 'FirstWord' of the drug name, if it does not match DO NOT list the summary under that drug.
- DO NOT summarize the notes of one drug into another drug summary. CHECK if the drug name present in the notes is the same as the drug you are placing the summary in. 
 
### Ensure that you followed these steps before giving response.
CHECKING STEP 1: did I include only the relevant?
CHECKING STEP 2: did I excluded all the unnecessary and irrelevant data?
CHECKING STEP 3: did I followed all the Instructions, Examples and format?
)
Generated Summary:"{summary}"
"""
)

loader1= TextLoader('input1.txt',encoding='utf-8')
input = loader1.load()

loader2= TextLoader('output1.txt',encoding='utf-8')
output = loader2.load()


parser = StrOutputParser()
chain = prompt1| model | parser 

result = chain.invoke({'note':input[0].page_content,'summary':output[0].page_content})
# parsed_result = json.loads(result)
print(result)
# print(parsed_result['truthfulness_score'])
# print(parsed_result['explanation'])

