from modulefinder import test
import os
from pydoc import text
from typing import Literal
from unittest import loader
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

class ChromadbManager:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="embedding-001")
        self.vector_store = Chroma(
            collection_name="test", 
            embedding_function=self.embeddings,
            persist_directory="./chroma_db_langchain.db"
            )

    def store(
        self,
        texts: list[str],
        ids: list[str],
        metadatas: list[dict]
        ):
        self.vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )

    def find(
        self,
        metadata: dict,
    ):
        result = self.vector_store.get(
            where=metadata,
            include=["embeddings", "documents"],
        )

        return result

    def query(
        self,
        query: str,
        metadata: dict,  
        k: int = 2  
    ): 
        ...

    def drop(
            self,
            metadata: dict
    ):
        ...
