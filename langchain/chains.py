import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import  PromptTemplate


load_dotenv()

class Answer (BaseModel):
    number_of_words: int

prompt_template = PromptTemplate.from_template("""
Te voy a dar el siguiente texto y quiero que me digas el numero que tiene.
Texto: {text}   
""")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
prompt = prompt_template.invoke({

    "text": "Este es un ejemplo con langchain"

})
llm_structured = llm.with_structured_output(Answer)

#llm_structured_response = llm_structured.invoke(prompt)
#print(llm_structured_response)

chain = prompt_template  llm_structured

chain_result = chain.invoke({

    "text": "Este es un ejemplo con langchain"

})

print(chain_result)