import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import  PromptTemplate

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

class AgentState(BaseModel):
    """
    Este estado contendra la informacion de la pregunta y respuesta y tambien del saludo del usuario
    """
    user_message: str= ""
    agent_response: str= ""
    greeting: bool = False


initial_state = AgentState(user_message = "Hola") 
initial_state.user_message += "!"
print(initial_state.user_message)

def greeting_node(agent_state: AgentState)-> AgentState:
    if "hola" in agent_state.user_message:
        agent_state.greeting = True
    else:
        agent_state.agent_response = False
    return agent_state

def response_node(agent_state: AgentState) -> AgentState:
    if agent_state.greeting:
        agent_state.agent_response = "¡Hola! ¿En qué puedo ayudarte hoy?"
    else:
        agent_state.agent_response = "Vuelve a intenrtarlo"
    return agent_state