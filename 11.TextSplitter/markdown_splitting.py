from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text = """📘 Project Name: Smart Student Tracker
A simple Python-based project to manage and track student data,

🔍 Features
Add new students with relevant info

View student details

Check if a student is passing

Easily extendable class-based design

🛠️ Tech Stack
Python 3.10+

No external dependencies"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size= 75,
    chunk_overlap = 0
)

result = splitter.split_text(text)

print(len(result))
print(result)