from typing import Annotated
from typing_extensions import List,Literal,TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode
import os
from dotenv import load_dotenv
import subprocess
import shlex
from langchain_core.tools import tool
from speech_to_text import get_text_from_speech
from pathlib import Path
from openai import OpenAI
import time

load_dotenv()



class State(TypedDict):
    messages:Annotated[list,add_messages]


# llm = ChatGoogleGenerativeAI(model=os.getenv("CHAT_MODEL_1"),google_api_key = os.getenv('GEMINI_API_KEY'))
llm = init_chat_model("gpt-4.1-2025-04-14", model_provider="openai")
@tool
def run_system_commands(command,state:State,cwd=None):
    """
    Runs System Commands by splitting the command string into arguments.
    Creates files relative to the current working directory of the Python script
    unless a different directory is specified using the 'cwd' parameter.

    Args:
        command: The command string to execute (e.g., 'ls -l /home', 'echo "hello" > output.txt').
        state: The current state object.
        cwd: Optional. The working directory for the command. If None, the
             command runs in the current working directory of the Python script.
             Provide a path string (e.g., '/tmp/my_output_dir').

    Returns:
        A string containing the exit code, STDOUT, and STDERR of the command.

    """
    cmd_list = shlex.split(command)
    result = subprocess.run(
        cmd_list,
        shell=False,
        cwd=cwd,
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
    while True:
        # start = "loop"
        # while start == "loop":
        #     start = input("press any key to continue.........")  

        # text = get_text_from_speech()
        text = input(" >:.:<   ")
        input_val = [{'role':'user','content':text}]

        for event in graph.stream({'messages':input_val}):
            for value in event.values():
                final_output = value['messages'][-1].content
                print('Assistant: ',value['messages'][-1].content)
                print(type(final_output))
        
        # if final_output:
            # text_to_speech(final_output)
        # else:
            # print("Error: No valid response received from LLM")


call_llm()

