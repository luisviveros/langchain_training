import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

#Como crear una plantilla para prompts

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

PromptTemplate = PromptTemplate.from_template( """

Me puedes traducir al idioma: {language}
la siguiente palabra: {word}

""")


#prompt = PromptTemplate.format(language="ingles", word="hola")

#prompt = PromptTemplate.invoke({
#    "language": "ingles",
#    "word": "carro"
#})

template = ChatPromptTemplate.from_messages([
    ("system", "Eres un traductor de idiomas"),
    ("user", "Me puedes traducir al idioma: {language} "),
    ("user", "la siguiente palabra: {word}")
])

# 1 era posicion: system o user 2da posicion: va el contenido

prompt = template.invoke({
    "language": "ingles",
    "word": "carro"
})  

llm_response = llm.invoke(prompt)

print(llm_response.content)