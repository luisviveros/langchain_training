from pydantic import BaseModel
from typing import Annotated, Literal
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langgraph.graph.messages import add_messages
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from chromadb_manager import ChromadbManager

load_dotenv()

class AgentState(BaseModel):
    user_message: str = ""
    query: str = ""
    context: str = ""
    language: Literal["es", "en"] = "es"
    messages: Annotated[list[AnyMessage], add_messages] = []
    question_type: Literal["appointment", "question"] = "question"

class LanguageOutput(BaseModel):
    lenguage: Literal["es", "en"]


class QuestionTypeOutput(BaseModel):
    question_type: Literal["appointment", "question"]

def detect_language_node(state: AgentState) -> LanguageOutput:
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.2)
    llm_parsed = llm.with_structured_output(LanguageOutput)
    response: LanguageOutput = llm_parsed.invoke(
        f"Detecta el lenguaje del siguiente texto, responde 'es' o 'en': '{state.user_message}' "
    )
    state.language = response.lenguage
    return response 


def detect_question_type_node(state: AgentState) -> Literal["appointment", "query_node"]:
    user_message = state.messages[-1].content
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.2)
    llm_parsed = llm.with_structured_output(QuestionTypeOutput)
    response: QuestionTypeOutput = llm_parsed.invoke(
        f"""
        Detecta si la pregunta del usuario es acerca de agendamiento de citas (appointment) o es una pregunta normal (question): 
        
        
        Pregunta del usuario:
        {user_message} 
        """
    )
    state.question_type = response.question_type
    if state.question_type == "appointment":
        return "appointment_node"
    else:
        return "query_node"
    
def appointment_node(state: AgentState)->AgentState:
    print("Cita agendada")
    state.messages = [AIMessage(content="Cita agendada para el usuario")]
    return state

def query_node(state: AgentState)->AgentState:
    history = state.messages[:-1]
    user_message = state.messages[-1]
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.2)
    response = llm.invoke(
        f"""
        Eres un agente que debe generar un query para que se realice una busqeuda vectorial,
        No debes agregar palabras que podrian ser causantes de que se busque en vectores no deseados.
        Debes usar el ultimo mensaje para generar el query y tambien puedes
        usar el historico para tener un mayor contexto de la pregunta del usuario.

        Historial de conversacion:
        {history}


        Nuevo mensaje del usuario:
        {user_message}

        Query
        """
    )
    state.query = response.content
    return state

def rag_node(state: AgentState)->AgentState:
    chromadb_manager = ChromadbManager()
    result = chromadb_manager.query(
        query=state.query,
        metadata={"filename": "consultorio.pdf"},
        k=2
    )
    state.context = "\n\n".join([doc.page_content for doc in result])
    return state

def response_node()