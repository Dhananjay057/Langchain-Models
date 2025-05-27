from pydantic import BaseModel, Field
from typing import List
import json

# Define the inner NoteSummary schema
class NoteSummary(BaseModel):
    noteDate: str = Field(..., title="Notedate", description="Date when the note was created.")
    summaryText: str = Field(..., title="Summarytext", description="Summary text of the note.")

# Define the main schema
class GenAndBillNotesSummary(BaseModel):
    rxNumber: str = Field(..., title="Rxnumber", description="Specify the prescription number (Rx) related to this task. Fetch the Rx Number only from the field **finalDisplayEntityId**.")
    drugName: str = Field(..., title="Drugname", description="Provide the name of the drug only from the field  **medicineName** associated with the Rx number.")
    diagnosis: List[NoteSummary] = Field(..., title="Bilnotessummary", description="Follow **Bil Notes Summary** Instructions to provide the summary.")
    currentPAstatus: str = Field(..., title="Currentpastatus", description="Provide the current PA status of the task. Fetch the PA Status only from the field **paStatus**.")
    LastTouchDate: str = Field(..., title="Lasttouchdate", description="Provide the date of the last touch for the task. Fetch the Last Touch Date only from the field **lastTouchDate**.")
    prescriberInformation: List[NoteSummary] = Field(..., title="Prescriberinformation", description="Follow **Prescriber Information** Instructions to provide the summary.")
    notesSummary: List[NoteSummary] = Field(..., title="Gennotessummary", description="Follow **Gen Notes Summary** Instructions to provide the summary.")
  
# Generate the JSON schema
schema = GenAndBillNotesSummary.schema_json(indent=2)

# Save to txt file
with open("gen_and_bill_notes_schema.txt", "w") as file:
    file.write(schema)

print("JSON schema saved to 'gen_and_bill_notes_schema.txt'")