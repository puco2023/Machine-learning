from langgraph.graph import StateGraph,END
from typing import TypedDict,List,Dict
import openai
import os
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langgraph.graph import StateGraph
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key = api_key)

llm = ChatOpenAI(
    model = "gpt-4o",temperature = 0
)
class StateAgent(TypedDict,total = False):
    query:str
    result : str
    more:bool
    history:List[Dict[str,str]]
def answer(state):
    history = state.get("history")
    response = client.chat.completions.create(
        model = "gpt-4o",
        messages=history
    )
    ans = response.choices[0].message.content
    state["history"].append({"role":"assistant","content":ans})
    state["result"] = ans
    print("AI: ",ans)
    return state
def ask(state):
    state["query"] = input("You: ")
    if not "history" in state:
        state["history"] = []
    state["history"].append({"role":"user","content":state["query"]})
    return state
def should_continue(state):
    if state["query"]=="end":
        return False
    else:
        return True
graph = StateGraph(StateAgent)
graph.add_node("ask",ask)
graph.add_node("answer",answer)
graph.add_node("should_continue",should_continue)
graph.set_entry_point("ask")
graph.add_conditional_edges(
    "ask",
    should_continue,
    {
        True:"answer",
        False:END
    }
)
graph.add_edge("answer","ask")

app = graph.compile()
app.invoke({})