from typing import TypedDict, Annotated, Sequence, Literal, List
import re

from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END, MessagesState, add_messages
from langchain_core.messages import RemoveMessage, AnyMessage, AIMessage

import requests
import time
from functools import lru_cache
import functools

def extract_python_code(text):
    pattern = r'```python\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

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

Make sure any generated graphs have at least one edge that leads to the END node - you need to define a stopping criteria!

You generate code using markdown python syntax, eg:

```python
...
```

Remember, only generate one of those code blocks!"""

gather_prompt = """You are tasked with helping build LangGraph applications. \
LangGraph is a framework for developing LLM applications. \
It represents agents as graphs. These graphs can contain cycles and often contain branching logic.

Your first job is to gather all the user requirements about the topology of the graph. \
You should have a clear sense of all the nodes of the graph/agent, and all the edges. 

You are conversing with a user. Ask as many follow up questions as necessary - but only ask ONE question at a time. \
Only gather information about the topology of the graph, not about the components (prompts, LLMs, vector DBs). \
If you have a good idea of what they are trying to build, call the `Build` tool with a detailed description.

Do not ask unnecessary questions! Do not ask them to confirm your understanding! The user will be able to \
correct you even after you call the Build tool, so just do enough to get an MVP."""

critique_prompt = """You are tasked with critiquing a junior developers first attempt at building a LangGraph application. \
Here is a long unit test file for LangGraph. This should contain a lot (but possibly not all) \
relevant information on how to use LangGraph.

<unit_test_file>
{file}
</unit_test_file>

Based on the conversation below, attempt to critique the developer. If it seems like the written solution is fine, then call the `Accept` tool.

Do NOT critique the internal logic of the nodes too much - just make sure the flow (the nodes and edges) are correct and make sense. \
It's totally fine to use dummy LLMs or dummy retrieval steps."""


class Build(TypedDict):
    requirements: str


class Accept(TypedDict):
    accept: bool

class AgentState(MessagesState):
    requirements: str
    code: str

def draft_answer(state: AgentState):
    github_url = "https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/tests/test_pregel.py"
    file_contents = load_github_file(github_url)
    messages = [
        {"role": "system", "content": prompt.format(file=file_contents)},
                   {"role": "user", "content": state.get('requirements')}
    ] + state['messages']
    model = ChatAnthropic(temperature=0, model_name="claude-3-5-sonnet-20240620")
    response = model.invoke(messages)
    return {"messages": [response]}


def gather_requirements(state: AgentState):
    messages = [
       {"role": "system", "content": gather_prompt}
   ] + state['messages']
    model = ChatAnthropic(temperature=0, model_name="claude-3-5-sonnet-20240620").bind_tools([Build])
    response = model.invoke(messages)
    if len(response.tool_calls) == 0:
        return {"messages": [response]}
    else:
        requirements = response.tool_calls[0]['args']['requirements']
        delete_messages = [RemoveMessage(id=m.id) for m in state['messages']]
        return {"requirements": requirements, "messages": delete_messages}


def _swap_messages(messages):
    new_messages = []
    for m in messages:
        if isinstance(m, AIMessage):
            new_messages.append({"role": "user", "content": m.content})
        else:
            new_messages.append({"role": "assistant", "content": m.content})
    return new_messages


error_parsing = """Make sure your response contains a code block in the following format:

```python
...
```

When trying to parse out that code block, got this error: {error}"""

def check(state: AgentState):
    last_answer = state['messages'][-1]
    try:
        code_blocks = extract_python_code(last_answer.content)
    except Exception as e:
        return {"messages": [{"role": "user", "content": error_parsing.format(error=str(e))}]}
    if len(code_blocks) == 0:
        return {"messages": [{"role": "user", "content": error_parsing.format(error="Did not find a code block!")}]}
    if len(code_blocks) > 1:
        return {"messages": [{"role": "user", "content": error_parsing.format(error="Found multiple code blocks!")}]}
    return {"code": code_blocks[0]}

def critique(state: AgentState):
    github_url = "https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/tests/test_pregel.py"
    file_contents = load_github_file(github_url)
    messages = [
                   {"role": "user", "content": critique_prompt.format(file=file_contents)},
                   {"role": "assistant", "content": state.get('requirements')},

               ] + _swap_messages(state['messages'])
    model = ChatAnthropic(temperature=0, model_name="claude-3-5-sonnet-20240620").bind_tools([Accept])
    response = model.invoke(messages)
    if len(response.tool_calls) == 0:
        return {"messages": [{"role": "user", "content": response.content}]}


def route_critique(state: AgentState) -> Literal["draft_answer", END]:
    if isinstance(state['messages'][-1], AIMessage):
        return END
    else:
        return "draft_answer"

def route_check(state: AgentState) -> Literal["critique", "draft_answer"]:
    if isinstance(state['messages'][-1], AIMessage):
        return "critique"
    else:
        return "draft_answer"


def route_start(state: AgentState) -> Literal["draft_answer", "gather_requirements"]:
    if state.get('requirements'):
        return "draft_answer"
    else:
        return "gather_requirements"


def route_gather(state: AgentState) -> Literal["draft_answer", END]:
    if state.get('requirements'):
        return "draft_answer"
    else:
        return END


# Define a new graph
workflow = StateGraph(AgentState, input=MessagesState)
workflow.add_node(draft_answer)
workflow.add_node(gather_requirements)
workflow.add_node(critique)
workflow.add_node(check)
workflow.set_conditional_entry_point(route_start)
workflow.add_conditional_edges("gather_requirements", route_gather)
workflow.add_edge("draft_answer", "check")
workflow.add_conditional_edges("check", route_check)
workflow.add_conditional_edges("critique", route_critique)
graph = workflow.compile()
