import os
from typing import Iterator
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

#numbers = [1,2,3,4,5]

#def numbers_generator(numbers: list[int]) -> Iterator[str]:
#    for number in numbers:
#        yield number  #devuelve un elemento a la vez
    
#try: 
#    for number in numbers_generator(numbers):
#        import time
#        time.sleep(0.10)
#        print(number, end = " ", flush=True)
#except Exception as e:
#    pass

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
llm_response = llm.invoke("que es un llm?")

for chunk in llm.stream("que es un llm?"):
    print(chunk.content, end = "", flush=True)