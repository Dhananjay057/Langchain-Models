from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader('Catalyst_optimiser.pdf')

docs= loader.load()

# print(docs[0].metadata)
print(docs[0].page_content)