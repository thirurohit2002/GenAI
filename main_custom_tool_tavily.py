# from dotenv import load_dotenv
# import os
# load_dotenv()
# from langchain.agents import create_agent
# from langchain.tools import tool
# from langchain_core.messages import HumanMessage #used to invoke the agent, i.e, input for agent exec
# from langchain_openai import ChatOpenAI
# from tavily import TavilyClient
#
# tavily = TavilyClient()
#
# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over internet
#     Args:
#         query: The query to search for
#     Returns:
#         The search result
#     """
#     print(f"Searching for {query}")
#     return tavily.search(query=query)
#
#
# llm = ChatOpenAI(openai_api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
# tools = [search]
# agent = create_agent(model=llm,tools=tools) #create_agent is autonomous agent which can/decides to use the tools provided to answer the user query
#
# def main():
#     print("Hello")
#     result = agent.invoke({"messages":[HumanMessage(content="search for 3 job postings for an java softwware engineer using RestAPI, Springboot in Bengaluru and Hyderabad on linkedin and list the details")]})
#     print(result)
#
# if __name__ == "__main__":
#      main()