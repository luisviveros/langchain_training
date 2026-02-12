import os
from typing import Literal
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import PromptTemplate

load_dotenv()

class GreetingOutput(BaseModel):
    greeting: bool

class LanguageOutput(BaseModel):
    language: Literal["es", "en"]

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
    prompt_template = PromptTemplate.from_template(
    """
    Debes detectar si en el siguiente texto el usuario esta saludando: {text}
    Tu respuesta debera ser true si detectas que el usuario esta saludando o 
    false si no esta saludando.
    """
    )
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    llm_parsed = llm.with_structured_output(GreetingOutput)
    chain = prompt_template | llm_parsed
    chain_response: GreetingOutput = chain.invoke({"text": agent_state.user_message})
    agent_state.greeting = chain_response.greeting
    return agent_state

def response_node(agent_state: AgentState) -> AgentState:
    if agent_state.greeting:
        agent_state.agent_response = "¡Hola! ¿En qué puedo ayudarte hoy?"
    else:
         llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
         llm_response = llm.invoke(agent_state.user_message)
         agent_state.agent_response = llm_response.content
    return agent_state

def evaluate_response(agent_state: AgentState) -> Literal["spanish_response_node", "english_response_node"]:

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    llm_parsed = llm.with_structured_output(LanguageOutput)
    llm_respone : LanguageOutput = llm_parsed.invoke(
        f"""
        En que lenguaje esta el siguiente texto, devuelve un json con el lenguaje 'es' en el caso de que 
        texto este en espanol o 'en' en el caso de que este en ingles
        Texto: {agent_state.agent_response}
        """
    )
    if llm_respone.language == "es":
        return "spanish_response_node"
    else:
        return "english_response_node"


def spanish_response_node(agent_state: AgentState) -> AgentState:
    agent_state.agent_response += "\nIdioma: Espanol"
    return agent_state

def english_response_node(agent_state: AgentState) -> AgentState:
    agent_state.agent_response += "\nIdioma: Ingles"
    return agent_state

graph = StateGraph(AgentState)
graph.add_node(greeting_node)
graph.add_node(response_node)
graph.add_node(spanish_response_node)
graph.add_node(english_response_node)

graph.add_edge(START, "greeting_node")
graph.add_edge("greeting_node", "response_node")
#graph.add_edge("response_node", END)
graph.add_conditional_edges("response_node", evaluate_response)
graph.add_edge("spanish_response_node", END)
graph.add_edge("english_response_node", END)

compiled_graph = graph.compile()

if __name__ == "__main__":
    initial_state = AgentState(user_message="Hola")
    response = compiled_graph.invoke(initial_state)
    print(response["agent_response"])