from typing import TypedDict, Annotated, Sequence, Literal

from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END, MessagesState

import requests
import time
from functools import lru_cache
import functools

# Cache duration in seconds (24 hours)
CACHE_DURATION = 24 * 60 * 60


def time_based_cache(seconds):
    def wrapper_cache(func):
        func = lru_cache(maxsize=None)(func)
        func.lifetime = seconds
        func.expiration = time.time() + func.lifetime

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            if time.time() >= func.expiration:
                func.cache_clear()
                func.expiration = time.time() + func.lifetime
            return func(*args, **kwargs)

        return wrapped_func

    return wrapper_cache


@time_based_cache(CACHE_DURATION)
def load_github_file(url):
    # Convert GitHub URL to raw content URL
    raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

    # Send a GET request to the raw URL
    response = requests.get(raw_url)

    # Check if the request was successful
    if response.status_code == 200:
        return response.text
    else:
        return f"Failed to load file. Status code: {response.status_code}"


prompt = """You are tasked with answering questions about LangGraph functionality and bugs.
Here is a long unit test file for LangGraph. This should contain a lot (but possibly not all) \
relevant information on how to use LangGraph.

<unit_test_file>
{file}
</unit_test_file>

Based on the information above, attempt to answer the user's questions. If you generate a code block, only \
generate a single code block - eg lump all the code together (rather than splitting up). \
You should encode helpful comments as part of that code block to understand what is going on. \
ALWAYS just generate the simplest possible example - don't make assumptions that make it more complicated. \
For "messages", these are a special object that looks like: {{"role": .., "content": ....}}

If users ask for a messages key, use MessagesState which comes with a built in `messages` key. \
You can import MessagesState from `langgraph.graph` and it is a TypedDict, so you can subclass it and add new keys to use as the graph state.

Make sure any generated graphs have at least one edge that leads to the END node - you need to define a stopping criteria!"""

def answer_question(state: MessagesState):
    github_url = "https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/tests/test_pregel.py"
    file_contents = load_github_file(github_url)
    messages = [
        {"role": "system", "content": prompt.format(file=file_contents)}
    ] + state['messages']
    model = ChatAnthropic(temperature=0, model_name="claude-3-5-sonnet-20240620")
    response = model.invoke(messages)
    return {"messages": [response]}

# Define a new graph
workflow = StateGraph(MessagesState)
workflow.add_node(answer_question)
workflow.set_entry_point("answer_question")
workflow.add_edge("answer_question", END)
graph = workflow.compile()
