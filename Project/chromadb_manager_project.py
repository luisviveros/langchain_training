import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

class ChromadbManager:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        self.vector_store = Chroma(
            collection_name="test",
            embedding_function=self.embeddings,
            persist_directory=os.path.join(_PROJECT_DIR, "chroma_db_langchain.db")
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
            include=["documents"],
        )

        return result["documents"]

    def query(
        self,
        query: str,
        metadata: dict,  
        k: int = 2  
    ): 
        result = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=metadata
        )
        
        return result




    def drop(
            self,
            metadata: dict
    ):
        self.vector_store.delete(
            where=metadata
            )
        
