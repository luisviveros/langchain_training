import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

llm = ChatAnthropic(model_name="claude-sonnet-4-20250514", temperature=0, timeout=None, stop=None)
llm_response = llm.invoke("que es un llm?")
print(llm_response.content)