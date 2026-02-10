import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from langchain_core.output_parsers import CommaSeparatedListOutputParser

load_dotenv()

class AnswerWithJustification(BaseModel):
    """Una respuesta para la pregunta con su justificacion"""
    answer: str
    justification: str

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
llm_structured = llm.with_structured_output(AnswerWithJustification)
llm_response = llm_structured.invoke("Cuantos años tengo si naci el 10 de Diciembre de 1998")
print(llm_response.justification)


#llm_response = llm.invoke("""
#        Devuelveme una lista de tres frutas,
#       El formato de la respuesta debera ser un string con los resultados separados
#       por coma.
#       Ejemplo de respuesta: manzana, uva, pera                           
#""")

#parser = CommaSeparatedListOutputParser()
#parser_result = parser.invoke(llm_response.content)
#print(parser_result)

# parser_result = parser.invoke("manzana, pera, naranja, uva")
# print(parser_result)