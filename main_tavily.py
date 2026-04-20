from dotenv import load_dotenv
import os
load_dotenv()
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage #used to invoke the agent, i.e, input for agent exec
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from typing import List
from pydantic import Field, BaseModel

class Source(BaseModel):
    """Scheme for source used by the agent"""

    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Scheme for agent response"""

    answer: str = Field(description="The content of the agent response")
    sources: List[Source] = Field(default_factory=list, description="List of sources used by the agent") #defaultFactory = Empty List


llm = ChatOpenAI(openai_api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
tools = [TavilySearch()]
agent = create_agent(model=llm,tools=tools, response_format=AgentResponse)

def main():
    print("Hello")
    result = agent.invoke({"messages":[HumanMessage(content="search for 3 job postings for an ai engineer using langchain in Bengaluru and Hyderabad on linkedin and list the details")]})
    print(result) #The result will be of type, Structured Response while using AgentResponse Data Structure(o/p)


if __name__ == "__main__":
    main()