import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END

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
    if "Hola" in agent_state.user_message:
        agent_state.greeting = True
    else:
        agent_state.greeting = False
    return agent_state

def response_node(agent_state: AgentState) -> AgentState:
    if agent_state.greeting:
        agent_state.agent_response = "¡Hola! ¿En qué puedo ayudarte hoy?"
    else:
        agent_state.agent_response = "Vuelve a intentarlo"
    return agent_state


graph = StateGraph(AgentState)
graph.add_node(greeting_node)
graph.add_node(response_node)

graph.add_edge(START, "greeting_node")
graph.add_edge("greeting_node", "response_node")
graph.add_edge("response_node", END)

compiled_graph = graph.compile()

initial_state = AgentState(user_message="Hola")
respone = compiled_graph.invoke(initial_state)
print(respone["agent_response"])