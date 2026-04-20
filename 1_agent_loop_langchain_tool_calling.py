from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model #helps to initialize a langchain chat model, with any model providers
from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage, ToolMessage #ToolMessage - contains result of Tool, SystemMessage - wrapper for system message to llm
from langsmith import traceable
load_dotenv()

MAX_ITERATION = 10
MODEL= 'openai/gpt-4o-mini'

@tool
def get_product_price(product: str) -> float:
    """Look for the price of the product in the catalog"""

    print(f"Looking for the price of {product} in the catalog...")
    prices={"laptop":1299.99, "headphones":149.95, "keyboard":89.50}
    return prices.get(product,0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply the discount to the price and return the final price
    Available discount tiers: bronze, silver and gold"""
    print(f"Executing apply_discount with price:{price} and discount_tier:{discount_tier}")
    discount_percentages = {"bronze":5, "silver":12, "gold":23}
    discount = discount_percentages.get(discount_tier,0)
    return round(price * (1-discount/100),2)

@traceable(name="LangChain agent loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tool_dict = {t.name: t for t in tools}

    llm = init_chat_model(MODEL, model_provider="openai", base_url="https://openrouter.ai/api/v1",
                          api_key=os.getenv("OPENROUTER_API_KEY"), temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")

    messages = [
        SystemMessage(content=(
            "You are a helpful shopping assistant."
            "You have access to a product catalog tool and a discount tool.\n\n"
            "STRICT RULES - you must follow thee exactly:\n"
            "1. NEVER guess or assume any product price."
            "You MUST have to call the get_product_price to get the real price.\n"
            "2. Only call apply discount AFTER you have received a price from the get_product_price. Pass the exact price "
            "returned by get_product_price to apply_discount, do NOT modify it in any way.\n"
            "3. NEVER calculate discounts yourself using math. "
            "Always use the apply_discount tool.\n"
            "4. If a user does not provide a discount_tier, ask them which tier to use - do NOT assume anything else"
        )
        ),
        HumanMessage(content=question)
    ]

    for iteration in range(1, MAX_ITERATION+1):
        print(f"---{iteration}---")
        ai_message = llm_with_tools.invoke(messages) #ai_message contains either the tool call information or the final ans
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"Tool to be called: {tool_name} with args: {tool_args}]")

        tool_to_use = tool_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")

        observation = tool_to_use.invoke(tool_args)

        print(f"Tool Result: {observation}")

        messages.append(ai_message)
        messages.append(ToolMessage(content=observation, tool_call_id=tool_call_id))
    print("Error: Max iterations reached, without final answer")
    return None

if __name__ == "__main__":
    print("LangChain Agent\n")
    result = run_agent("What is the price of the laptop product after applying a gold discount tier?")

