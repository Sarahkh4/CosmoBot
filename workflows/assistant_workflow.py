from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage

def create_workflow(llm, tools):
    """Create and configure the LangGraph workflow"""
    builder = StateGraph(MessagesState)
    
    # Define nodes
    builder.add_node("assistant", _assistant_node(llm))
    builder.add_node("tools", ToolNode(tools))
    
    # Define edges
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    
    return builder.compile(checkpointer=MemorySaver())

def _assistant_node(llm):
    """Create assistant node with system prompt"""
    sys_msg = SystemMessage(content=SYSTEM_PROMPT)
    
    def assistant(state: MessagesState):
        return {"messages": [llm.bind_tools(tools).invoke(
            [sys_msg] + state["messages"][-10:]
        )]}
    
    return assistant