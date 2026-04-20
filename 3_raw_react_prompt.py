import re # parse row response from the llm which will be text now, and we are not going to get formatting(JSON Schema) from function calling
import inspect # to get metadata from functions and propagate to llm

from openai import OpenAI
from dotenv import load_dotenv
import os
from langsmith import traceable
load_dotenv()

MAX_ITERATION = 10
MODEL= 'openai/gpt-4o-mini'

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look for the price of the product in the catalog"""
    print(f"Looking for the price of {product} in the catalog...")
    prices={"laptop":1299.99, "headphones":149.95, "keyboard":89.50}
    return prices.get(product,0)

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply the discount to the price and return the final price
    Available discount tiers: bronze, silver and gold"""
    print(f"Executing apply_discount with price:{price} and discount_tier:{discount_tier}")
    price = float(price)
    discount_percentages = {"bronze":5, "silver":12, "gold":23}
    discount = discount_percentages.get(discount_tier,0)
    return round(price * (1-discount/100),2)

tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
}

def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        original_function = getattr(tool_function, "__wrapped__",tool_function) # __wrapped__ is used to bypass the decorator, its used here in order to get metadata of the func before @traceable
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function)
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions) #joining all the desc as single string

tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

@traceable(name="OpenAPI chat", run_type="llm")
def openai_chat_traced(model, messages, stop=None, temperature=0):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        stop=stop,
        temperature=temperature,
    )

@traceable(name="OpenAI Agent Loop")
def run_agent(question: str):
    print(f"Question: {question}")
    print("------")

    prompt = react_prompt.format(question=question)
    scratchpad = ""

    for iteration in range(1, MAX_ITERATION + 1):
        print(f"\nIteration {iteration}")
        full_prompt = prompt + scratchpad

        # Stop token prevents the LLM from generating its own Observation, halucination
        # we inject the real tool result instead in the upcoming iterations with the hellp of scratchpad
        response = openai_chat_traced(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            stop=["\nObservation:"],
            temperature=0,
        )
        output = response.choices[0].message.content or ""
        print(f"LLM Output:\n{output}")

        print("Looking for Final Answer in LLM output")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"Final Answer: {final_answer}")
            print("\n" + "-------------------")
            print(f"Final Answer: {final_answer}")
            return final_answer

        print("Looking for Action and Action Input in LLM output...")

        action_match = re.search(r"Action:\s*(.+)", output) #get_product_price
        action_input_match = re.search(r"Action Input:\s*(.+)", output) #product=laptop

        if not action_match or not action_input_match:
            print(
                "ERROR: Could not parse Action/Action Input from LLM output"
            )
            break

        print(f"Action match: {action_match}, Action Input Match: {action_input_match}")
        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip() #e.g., product=laptop

        print(f"Tool Selected -> {tool_name} with args: {tool_input_raw}")

        # Split comma-separated args; strip key= prefix if LLM outputs key=value format
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args] #e.g., just laptop

        print(f"Tool Executing -> {tool_name}({args})...")
        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list(tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))

        print(f"  [Tool Result] {observation}")

        #History is one growing string re-sent every iteration (replaces messages.append)
        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    print("ERROR: Max iterations reached without a final answer")
    return None

if __name__ == "__main__":
    print("Raw ReAct Agent\n")
    result = run_agent("What is the price of the laptop product after applying a gold discount tier?")

