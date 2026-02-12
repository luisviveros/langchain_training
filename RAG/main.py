import os
from typing import Literal
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

