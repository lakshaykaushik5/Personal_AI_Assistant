from typing import Annotated
from typing_extensions import List,Literal,TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode
import os
from dotenv import load_dotenv
import subprocess
import shlex
from langchain_core.tools import tool

load_dotenv()

class State(TypedDict):
    messages:Annotated[list,add_messages]


llm = init_chat_model("gpt-4.1-2025-04-14",model_provider="openai")

@tool
def run_system_commands(command,state:State):
    """Runs System Commands by splitting the command string into arguments.
       Example: command='ls -l /home'
    """
    cmd_list = shlex.split(command)
    result = subprocess.run(
        cmd_list,
        shell=False,
        capture_output=True,
        text=True,
        check=False
    )
    output = f"Exit Code: {result.returncode}\n"
    if result.stdout:
        output += f"STDOUT:\n{result.stdout}\n"
    if result.stderr:
        output += f"STDERR:\n{result.stderr}\n"
    return output


def should_continue(state:State)->Literal["tools","__end__"]:
    """Determines weather to continue with tool or end"""

    last_message = state['messages'][-1]
    if isinstance(last_message,AIMessage) and last_message.tool_calls:
        print("---using--tools---")
        return 'tools'
    return "__end__"

tools = [run_system_commands]
tool_node = ToolNode(tools)

llm_with_tools = llm.bind_tools(tools)

def chatbot(state:State):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}

graph_builder = StateGraph(State)

graph_builder.add_node('chatbot',chatbot)
graph_builder.add_node('tools',tool_node)
graph_builder.add_edge(START,'chatbot')
graph_builder.add_conditional_edges('chatbot',should_continue)
graph_builder.add_edge('tools','chatbot')

graph_builder.add_edge('chatbot',END)

graph = graph_builder.compile()

def call_llm():
    input_val = [{'role':'user','content':'create a file name requirement.txt and pip freeze all the install packages in it do it in current working directory and also run pwd command afther all the operation'}]

    for event in graph.stream({'messages':input_val}):
        for value in event.values():
            print('Assistant: ',value['messages'][-1].content)

call_llm()