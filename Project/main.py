import sys
import os
import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'RAG_LangChain'))

from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Annotated, Literal
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langgraph.graph import add_messages
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from chromadb_manager import ChromadbManager
from langgraph.graph import StateGraph, START, END
from fastapi import FastAPI
from langgraph.checkpoint.memory import MemorySaver
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/calendar"]
CREDENTIALS_FILE = os.path.join(os.path.dirname(__file__), "credentials.json")
TOKEN_FILE = os.path.join(os.path.dirname(__file__), "token.json")

def get_google_calendar_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
    return build("calendar", "v3", credentials=creds)

class AgentState(BaseModel):
    user_message: str = ""
    query: str = ""
    context: str = ""
    language: Literal["es", "en"] = "es"
    messages: Annotated[list[AnyMessage], add_messages] = []
    question_type: Literal["appointment", "question"] = "question"

class LanguageOutput(BaseModel):
    lenguage: Literal["es", "en"]


class QuestionType(BaseModel):
    question_type: Literal["appointment", "question"]

def create_appointment(date: str, description: str = ""):
    """
    Recibe una fecha (formato: YYYY-MM-DD o DD/MM/YYYY) y opcionalmente una descripción
    para agendar una cita médica en Google Calendar.
    """
    try:
        # Normalizar formato de fecha
        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
            try:
                parsed_date = datetime.datetime.strptime(date.strip(), fmt)
                break
            except ValueError:
                continue
        else:
            return f"No pude interpretar la fecha '{date}'. Por favor usa formato DD/MM/YYYY."

        event = {
            "summary": description if description else "Cita médica",
            "description": description,
            "start": {
                "date": parsed_date.strftime("%Y-%m-%d"),
                "timeZone": "America/Bogota",
            },
            "end": {
                "date": parsed_date.strftime("%Y-%m-%d"),
                "timeZone": "America/Bogota",
            },
        }

        service = get_google_calendar_service()
        created_event = service.events().insert(calendarId="primary", body=event).execute()
        event_link = created_event.get("htmlLink", "")
        return f"Cita médica agendada para el {parsed_date.strftime('%d/%m/%Y')}. Puedes verla en: {event_link}"

    except Exception as e:
        return f"No se pudo agendar la cita en Google Calendar: {str(e)}"



def detect_language_node(state: AgentState) -> LanguageOutput:
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.2)
    llm_parsed = llm.with_structured_output(LanguageOutput)
    response: LanguageOutput = llm_parsed.invoke(
        f"Detecta el lenguaje del siguiente texto, responde 'es' o 'en': '{state.user_message}' "
    )
    state.language = response.lenguage
    print(f"Lenguaje detectado: {state.language}")

    return state 


def detect_question_type_node(state: AgentState) -> Literal["appointment_node", "query_node"]:
    user_message = state.messages[-1].content
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.2)
    llm_parsed = llm.with_structured_output(QuestionType)
    response: QuestionType = llm_parsed.invoke(
        f"""
        Clasifica la intención del usuario en una de estas dos categorías:

        - "appointment": El usuario quiere AGENDAR, RESERVAR o PROGRAMAR una cita.
          Ejemplos: "Quiero agendar una cita", "Reservar para el lunes", "Necesito una cita para el 17/03".

        - "question": El usuario quiere INFORMACIÓN sobre el consultorio.
          Ejemplos: precios, costos, tarifas, servicios disponibles, especialidades, horarios,
          requisitos, ubicación, o cualquier pregunta de consulta general.
          También aplica para preguntas de seguimiento cortas como "Y de cardiología?", "¿Y medicina interna?".

        IMPORTANTE: Mencionar una especialidad médica NO significa que quiere agendar una cita.
        Solo clasifica como "appointment" si el usuario explícitamente quiere AGENDAR.

        Pregunta del usuario:
        {user_message}
        """
    )
    state.question_type = response.question_type
    print(f"Tipo de pregunta detectada: {state.question_type}")
    if state.question_type == "appointment":
        return "appointment_node"
    else:
        return "query_node"
    
def appointment_node(state: AgentState)->AgentState:
    tools = [create_appointment]
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.2).bind_tools(tools)
    response = llm.invoke(state.messages)
    if response.tool_calls:
        function_result = None
        for tool_call in response.tool_calls:
            if tool_call["name"] == "create_appointment":
                args = tool_call["args"]
                function_result = create_appointment(**args)
        state.messages = [AIMessage(content=function_result)]
    else:
        state.messages = [AIMessage(content=response.content)]
    return state

def query_node(state: AgentState)->AgentState:
    history = state.messages[:-1]
    user_message = state.messages[-1]
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.2)
    response = llm.invoke(
        f"""
        Eres un agente que debe generar un query para realizar una búsqueda vectorial en documentos de un consultorio médico.

        IMPORTANTE: Si el nuevo mensaje del usuario es una pregunta corta o de seguimiento (como "Y de cardiologia?", "¿Y el precio?"),
        debes enriquecer el query combinando la intención del historial con el nuevo mensaje.
        Por ejemplo: si antes se preguntó por "precio consulta general" y ahora dice "Y de cardiologia",
        el query debe ser "precio consulta cardiologia".

        Historial de conversacion:
        {history}

        Nuevo mensaje del usuario:
        {user_message}

        Genera SOLO el query de búsqueda, sin explicaciones adicionales.
        Query:
        """
    )
    state.query = response.content
    print(f"Query generado: {state.query}")
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

def response_node(state: AgentState)->AgentState:
    history = state.messages[:-1]
    user_message = state.messages[-1]
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.2)
    llm_response = llm.invoke(
        f"""
        Eres un asistente que responde preguntas de los usuarios  relacionadas a consultorio medico,
        usa la información del contexto y el historial para responder.

        Contexto:
        {state.context}

        Historial de conversacion:
        {history}


        Pregunta del usuario:
        {user_message}

        """
    )
    new_message = AIMessage(content=llm_response.content)
    state.messages = [new_message]
    return state

app = FastAPI()


graph = StateGraph(AgentState)
graph.add_node(detect_language_node)
graph.add_node(appointment_node)
graph.add_node(query_node)
graph.add_node(rag_node)
graph.add_node(response_node)

graph.add_edge(START, "detect_language_node")
graph.add_conditional_edges("detect_language_node", detect_question_type_node)
graph.add_edge("appointment_node", END)
graph.add_edge("query_node", "rag_node")
graph.add_edge("rag_node", "response_node")
graph.add_edge("response_node", END)

initial_state = AgentState()
checkpointer = MemorySaver()
compiled_graph = graph.compile(checkpointer=checkpointer)

#_ = compiled_graph.get_graph().draw_mermaid_png(output_file_path="agent.png")
class AgentInput(BaseModel):
    question: str

@app.post("/run_")
def run(agent_input: AgentInput):
    #question = "Quiero agendar una cita medica para el dia 17/03/2026 a las 10 am"
    user_message = HumanMessage(content=agent_input.question)
    initial_state.messages = [user_message]
    agent_response = compiled_graph.invoke(
        initial_state,
        config ={"configurable": {"thread_id": 1}}
        )
    print(agent_response["messages"][-1].content)
    return JSONResponse(content=agent_response["messages"][-1].content)
