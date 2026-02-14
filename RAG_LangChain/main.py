from modulefinder import test
import os
from pydoc import text
from typing import Literal
from unittest import loader
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter
from uuid_utils import uuid4
from chromadb_manager import ChromadbManager

load_dotenv()

loader = PyPDFLoader("files/lista_productos.pdf")
text = ""
content = loader.load()
# print(content)

for page in content:
    text += page.page_content + "\n"

#Separadores de texto

#text_splitter = CharacterTextSplitter(
#    separator="\n",
#    chunk_size=1000,
#)

#text_splitter = RecursiveCharacterTextSplitter(
#    separators=["\n\n", "\n", " ", ""],
#    chunk_size=1000,
#    chunk_overlap=200,
#)

text_splitter = TokenTextSplitter(
    chunk_size=12,
    chunk_overlap=5,
)

texts = text_splitter.split_text(text)
#print(texts)


chromadb_manager = ChromadbManager()

uuids = [str(uuid4()) for _ in range(len(texts))]
metadatas = [{"filename": "lista_productos.pdf"} for _ in range(len(texts))]

#chromadb_manager.store(
#    texts=texts,
#    ids=uuids,
#    metadatas=metadatas
#    )

#print(chromadb_manager.find({"filename": "lista_productos.pdf"}))
query = "Cual es el precio de la silla ergonomica herman miller"
result = chromadb_manager.query(
    query=query,
    metadata={"filename": "lista_productos.pdf"},
    k=2
)

context = "\n\n".join([doc.page_content for doc in result])

print(context)