import os
from uuid import uuid4
from chromadb_manager_project import ChromadbManager
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from dotenv import load_dotenv
load_dotenv()

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

chromadb_manager = ChromadbManager()
loader = PyPDFLoader(os.path.join(_PROJECT_DIR, "consultorio.pdf"))

text = ""
for doc in loader.load():
    text += doc.page_content + "\n"

text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=50, model_name="gpt-4o-mini")
texts = text_splitter.split_text(text)
uuids = [str(uuid4()) for _ in range(len(texts))]

#metadatas = [{"filename": "consultorio.pdf"} for _ in range(len(texts))]

#chromadb_manager.store(
#    texts=texts,
#    metadatas=metadatas,
#    ids=uuids
#)

response = chromadb_manager.query(
    query="Quiero saber los precios de la consulta",
    metadata ={"filename": "consultorio.pdf"}
)

print(response)