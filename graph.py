"""LangGraph setup and flow configuration"""
import sys
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from core import State
from agents import (
    classify_intent,
    extract_product_metadata,
    ask_clarification,
    faq_assistant,
    product_assistant,
    route_by_intent,
    route_by_metadata,
)
from services import get_tools

def build_graph():
    """Build graph with 7 nodes and conditional routing"""
    print("Building graph...")
    
    # Initialize graph builder
    builder = StateGraph(State)
    
    # Add nodes - ek ek node graph me add karo
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("extract_product_metadata", extract_product_metadata)
    builder.add_node("ask_clarification", ask_clarification)
    builder.add_node("faq_assistant", faq_assistant)
    builder.add_node("product_assistant", product_assistant)
    builder.add_node("tools", ToolNode(get_tools()))
    
    # Add edges
    builder.add_edge(START, "classify_intent")
    
    # Route by intent
    builder.add_conditional_edges(
        "classify_intent",
        route_by_intent
    )
    
    # Route by metadata
    builder.add_conditional_edges(
        "extract_product_metadata",
        route_by_metadata
    )
    
    # Clarification ends (wait for user response)
    builder.add_edge("ask_clarification", END)
    
    # FAQ ReAct loop
    builder.add_conditional_edges("faq_assistant", tools_condition)
    builder.add_edge("tools", "faq_assistant")
    
    # Product ReAct loop
    builder.add_conditional_edges("product_assistant", tools_condition)
    builder.add_edge("tools", "product_assistant")
    
    # Compile with memory
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
   
 # uncomment the below lines if u wnat to see the png of the langgraph
    # with open("graph.png", "wb") as f:
    #      f.write(graph.get_graph().draw_mermaid_png())

    print("Graph built successfully")
    return graph


# if __name__ == "__main__":
#     build_graph()