import os
import datetime
from zoneinfo import ZoneInfo

TZ_MEXICO = ZoneInfo("America/Mexico_City")

from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Annotated, Literal, cast
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, ToolMessage
from langgraph.graph import add_messages
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from chromadb_manager_project import ChromadbManager
from langgraph.graph import StateGraph, START, END
from fastapi import FastAPI
from langgraph.checkpoint.memory import MemorySaver
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

load_dotenv()

SERVICES_INFO = {
    "consulta general":        {"label": "Consulta General",                    "cost": "$50"},
    "pediatria":               {"label": "Pediatría",                           "cost": "$60"},
    "pediatría":               {"label": "Pediatría",                           "cost": "$60"},
    "ginecologia":             {"label": "Ginecología",                         "cost": "$70"},
    "ginecología":             {"label": "Ginecología",                         "cost": "$70"},
    "cardiologia":             {"label": "Cardiología",                         "cost": "$80"},
    "cardiología":             {"label": "Cardiología",                         "cost": "$80"},
    "medicina interna":        {"label": "Medicina Interna",                    "cost": "$75"},
    "examenes de laboratorio": {"label": "Exámenes de Laboratorio",             "cost": "Desde $30 (dependiendo de las pruebas)"},
    "exámenes de laboratorio": {"label": "Exámenes de Laboratorio",             "cost": "Desde $30 (dependiendo de las pruebas)"},
    "vacunacion":              {"label": "Vacunación",                          "cost": "Desde $20 (según el tipo de vacuna)"},
    "vacunación":              {"label": "Vacunación",                          "cost": "Desde $20 (según el tipo de vacuna)"},
    "control de diabetes":     {"label": "Control de Diabetes e Hipertensión",  "cost": "$40 por consulta"},
    "diabetes e hipertension": {"label": "Control de Diabetes e Hipertensión",  "cost": "$40 por consulta"},
    "diabetes e hipertensión": {"label": "Control de Diabetes e Hipertensión",  "cost": "$40 por consulta"},
}

# weekday(): 0=Lunes, 6=Domingo. Tupla (hora_inicio, hora_fin) en formato 24h
WORKING_HOURS = {
    0: (8, 19),   # Lunes
    1: (8, 19),   # Martes
    2: (8, 19),   # Miércoles
    3: (8, 19),   # Jueves
    4: (8, 19),   # Viernes
    5: (9, 14),   # Sábado
    6: None,      # Domingo: cerrado
}

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

def check_availability(date: str) -> str:
    """
    Consulta los horarios disponibles para una fecha específica.
    Revisa Google Calendar para evitar empalmes y respeta el horario del consultorio:
    Lunes a Viernes 8:00 AM - 7:00 PM, Sábado 9:00 AM - 2:00 PM, Domingo cerrado.
    """
    try:
        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
            try:
                parsed_date = datetime.datetime.strptime(date.strip(), fmt)
                break
            except ValueError:
                continue
        else:
            return f"No pude interpretar la fecha '{date}'. Por favor usa formato DD/MM/YYYY."

        weekday = parsed_date.weekday()
        day_names = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
        day_name = day_names[weekday]

        if WORKING_HOURS[weekday] is None:
            return f"El consultorio está cerrado los domingos. Por favor elige otro día."

        start_hour, end_hour = WORKING_HOURS[weekday]

        # Consultar eventos existentes en Google Calendar
        calendar_service = get_google_calendar_service()
        time_min = parsed_date.strftime("%Y-%m-%d") + "T00:00:00-06:00"
        time_max = parsed_date.strftime("%Y-%m-%d") + "T23:59:59-06:00"
        events_result = calendar_service.events().list(
            calendarId="primary",
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime"
        ).execute()
        events = events_result.get("items", [])

        # Marcar horas ocupadas (convertir a hora local para comparar correctamente)
        occupied_hours = set()
        for event in events:
            start = event["start"].get("dateTime", event["start"].get("date", ""))
            if "T" in start:
                event_start = datetime.datetime.fromisoformat(start).astimezone(TZ_MEXICO)
                occupied_hours.add(event_start.hour)

        # Generar slots disponibles
        available_slots = []
        for hour in range(start_hour, end_hour):
            if hour not in occupied_hours:
                slot = datetime.time(hour, 0).strftime("%I:%M %p")
                available_slots.append(slot)

        if not available_slots:
            return (
                f"No hay horarios disponibles para el {day_name} {parsed_date.strftime('%d/%m/%Y')}. "
                f"Todos los espacios están ocupados. Por favor elige otra fecha."
            )

        schedule = "8:00 AM - 7:00 PM" if weekday < 5 else "9:00 AM - 2:00 PM"
        slots_str = " | ".join(available_slots)
        return (
            f"📅 Horarios disponibles para el {day_name} {parsed_date.strftime('%d/%m/%Y')} "
            f"(horario de atención: {schedule}):\n{slots_str}\n\n"
            f"¿A qué hora prefieres tu cita?"
        )

    except Exception as e:
        return f"No pude verificar la disponibilidad: {str(e)}"


def create_appointment(date: str, service: str = "", time: str = "", description: str = ""):
    """
    Agenda una cita médica en Google Calendar.
    - date: fecha en formato DD/MM/YYYY o YYYY-MM-DD
    - service: tipo de servicio médico (ej: 'consulta general', 'pediatria', 'cardiologia', etc.)
    - time: hora opcional en formato HH:MM o H:MM am/pm (ej: '10:00', '2:30 pm')
    - description: descripción adicional opcional
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

        # Resolver servicio y costo
        service_key = service.strip().lower() if service else ""
        service_data = SERVICES_INFO.get(service_key, None)
        service_label = service_data["label"] if service_data else (service.strip() if service else "Cita médica")
        service_cost = service_data["cost"] if service_data else None

        event_summary = service_label
        event_description = description if description else service_label

        if time:
            # Normalizar y parsear la hora
            time_clean = time.strip().lower()
            parsed_time = None
            for fmt in ("%I:%M %p", "%I %p", "%H:%M", "%H"):
                try:
                    parsed_time = datetime.datetime.strptime(time_clean, fmt)
                    break
                except ValueError:
                    continue
            if parsed_time is None:
                return f"No pude interpretar la hora '{time}'. Por favor usa formato HH:MM o H:MM am/pm."

            start_dt = parsed_date.replace(hour=parsed_time.hour, minute=parsed_time.minute, second=0)
            end_dt = start_dt + datetime.timedelta(hours=1)

            # Verificar que el slot no esté ocupado antes de crear
            day_min = parsed_date.strftime("%Y-%m-%d") + "T00:00:00-06:00"
            day_max = parsed_date.strftime("%Y-%m-%d") + "T23:59:59-06:00"
            existing_events = get_google_calendar_service().events().list(
                calendarId="primary",
                timeMin=day_min,
                timeMax=day_max,
                singleEvents=True,
            ).execute().get("items", [])
            for existing in existing_events:
                ev_start_str = existing["start"].get("dateTime", "")
                if "T" in ev_start_str:
                    ev_local = datetime.datetime.fromisoformat(ev_start_str).astimezone(TZ_MEXICO)
                    if ev_local.hour == start_dt.hour:
                        date_label_err = parsed_date.strftime("%d/%m/%Y")
                        time_label_err = start_dt.strftime("%I:%M %p")
                        return (
                            f"⚠️ El horario de las {time_label_err} del {date_label_err} ya está ocupado. "
                            f"Por favor elige otra hora disponible."
                        )

            event = {
                "summary": event_summary,
                "description": event_description,
                "start": {
                    "dateTime": start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                    "timeZone": "America/Mexico_City",
                },
                "end": {
                    "dateTime": end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                    "timeZone": "America/Mexico_City",
                },
            }
            time_label = start_dt.strftime("%I:%M %p")
        else:
            event = {
                "summary": event_summary,
                "description": event_description,
                "start": {
                    "date": parsed_date.strftime("%Y-%m-%d"),
                    "timeZone": "America/Mexico_City",
                },
                "end": {
                    "date": parsed_date.strftime("%Y-%m-%d"),
                    "timeZone": "America/Mexico_City",
                },
            }
            time_label = None

        calendar_service = get_google_calendar_service()
        created_event = calendar_service.events().insert(calendarId="primary", body=event).execute()
        event_link = created_event.get("htmlLink", "")
        date_label = parsed_date.strftime("%d/%m/%Y")

        time_part = f" a las {time_label}" if time_label else ""
        cost_part = f"\n💰 Costo: {service_cost}" if service_cost else ""

        return (
            f"✅ Tu cita de {service_label} ha sido agendada para el {date_label}{time_part}."
            f"{cost_part}"
            f"\n⏰ Por favor trata de llegar 10 minutos antes al consultorio."
            f"\n🔗 Puedes verla en: {event_link}"
        )

    except Exception as e:
        return f"No se pudo agendar la cita en Google Calendar: {str(e)}"



def detect_language_node(state: AgentState) -> AgentState:
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.2)
    llm_parsed = llm.with_structured_output(LanguageOutput)
    response = cast(LanguageOutput, llm_parsed.invoke(
        f"Detecta el lenguaje del siguiente texto, responde 'es' o 'en': '{state.user_message}' "
    ))
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
    
def appointment_node(state: AgentState) -> AgentState:
    tool_map = {
        "check_availability": check_availability,
        "create_appointment": create_appointment,
    }
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.2).bind_tools(
        list(tool_map.values())
    )
    messages = list(state.messages)

    for _ in range(5):  # máximo 5 iteraciones para evitar loops infinitos
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            state.messages = [AIMessage(content=response.content)]
            break

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            if tool_name in tool_map:
                result = tool_map[tool_name](**tool_call["args"])
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

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
