from itertools import chain
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
from langchain_core.prompts import PromptTemplate

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
    chunk_size=500,
    chunk_overlap=50,
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

#print(len(chromadb_manager.find({"filename": "lista_productos.pdf"})))


query = "Cual es el precio de la silla ergonomica herman miller"
result = chromadb_manager.query(
    query=query,
    metadata={"filename": "lista_productos.pdf"},
    k=2
)

context = "\n\n".join([doc.page_content for doc in result])

#print(context)

prompt_template = PromptTemplate.from_template(
    """
    Eres un agente que responde preguntas acerca de productos tecnologicos.
    Debes responer en base a la informacion que existe en el 'Contexto'.

    *Contexto*:
    {context}

    *Pregunta del usuario*:
    {query}
    """
)

llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.2)
chain = prompt_template | llm
chain_result = chain.invoke({
    "context": context,
    "query": query
})

print(chain_result.content)